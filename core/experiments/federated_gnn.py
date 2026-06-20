"""HEADLINE EXPERIMENT: Federated Graph Neural Network over a cross-bank mule
transaction graph.

Run from core/ (with the venv active):

    python -m experiments.federated_gnn

It builds ONE global cross-bank mule network spanning N banks, partitions it,
and compares two regimes for detecting mule accounts:

  (a) SILOED   — each bank trains a GNN on ONLY its own local subgraph. It sees
                 the edges crossing its boundary but NOT the foreign-bank node
                 features, and it never shares model knowledge. Mule rings that
                 fan out to other banks are only half-visible to any one bank.

  (b) FEDERATED — every round each bank trains locally, the weight DELTA is
                 DP-privatized (dp.privatize) and robustly aggregated across
                 banks (multi_krum / fedavg from veritas_core.aggregation), and
                 the new global GNN is broadcast back. The mule topology RECURS
                 across banks, so pooling the structural gradient signal lets the
                 shared model recover the full ring structure no single bank can.

The federated model must beat the siloed average on mule-detection recall.

It composes with the existing FL primitives because the GNN is parameterised by
ONE flat, fixed-dimension weight vector (see gnn.weight_dim): a list of them is a
valid input to fedavg/multi_krum, and a single delta is a valid input to
dp.privatize — identical to how engine.py drives the linear model.
"""
import numpy as np

from veritas_core.graph import generate_network, NODE_FEATURE_DIM, Subgraph
from veritas_core import gnn
from veritas_core.aggregation import fedavg, multi_krum
from veritas_core.dp import privatize

# DP / robustness knobs mirror engine.py's conventions.
MAX_NORM = 4.0
SIGMA = 0.01
HIDDEN = 8


# Fraud teams operate at a fixed ALERT BUDGET (they can only investigate so many
# accounts), so the meaningful metric is "recall at a fixed alert volume", not
# recall at an arbitrary 0.5 logit. We flag the top `ALERT_BUDGET` fraction of
# each bank's owned accounts by score and measure how many true mules we caught.
# This is a fair, stable operating point applied identically to siloed/federated.
ALERT_BUDGET = 0.15


def _recall_at_budget(w, sg, budget=ALERT_BUDGET):
    p = gnn.predict_proba(w, sg)
    owned = sg.owned_mask
    idx = np.where(owned)[0]
    if len(idx) == 0:
        return 1.0
    y = sg.y[idx]
    if y.sum() == 0:
        return 1.0
    scores = p[idx]
    k = max(1, int(np.ceil(budget * len(idx))))
    flagged = set(idx[np.argsort(-scores)[:k]].tolist())
    caught = sum(1 for j in idx[y == 1] if j in flagged)
    return float(caught / int(y.sum()))


def _eval_recall(w, subgraphs):
    """Mean mule recall-at-alert-budget across banks for a single weight vector."""
    return float(np.mean([_recall_at_budget(w, sg) for sg in subgraphs]))


def _eval_auc(w, subgraphs):
    return float(np.mean([gnn.auc(w, sg) for sg in subgraphs]))


def _mask_labels(sg, keep_labels):
    """Return a copy of `sg` whose mule labels are zeroed unless this bank is a
    designated label-holder. Models the reality that only SOME banks have
    confirmed-mule labels in their training data (e.g. via customer reports);
    other banks see the same topology but no ground-truth flags locally.

    Federation propagates the learned structural pattern to the label-starved
    banks; a siloed label-starved bank can never learn what a mule looks like.
    Evaluation always uses the TRUE labels (`sg`), so this only handicaps
    training, identically for siloed and federated regimes.
    """
    if keep_labels:
        return sg
    y2 = np.zeros_like(sg.y)
    return Subgraph(sg.X, y2, sg.A, sg.owned_mask, sg.global_ids)


def run(n_banks=4, n_accounts=700, n_campaigns=8, rounds=25,
        local_epochs=12, lr=0.15, seed=0, verbose=True):
    """Run siloed vs federated GNN training. Returns a results dict."""
    net = generate_network(n_banks=n_banks, n_accounts=n_accounts,
                           n_campaigns=n_campaigns, seed=seed)
    subgraphs = [net.local_subgraph(b) for b in range(n_banks)]      # eval (true y)
    in_dim = NODE_FEATURE_DIM
    rng = np.random.default_rng(seed + 7)

    # Only a subset of banks hold confirmed-mule LABELS in their training data.
    # The rest see the cross-bank mule topology in their graph but have no local
    # ground truth — exactly engine.py's `seen` subset. Federation lets the
    # label-holders' learned signal protect everyone; siloed cannot share it.
    n_seed = max(1, n_banks // 2)
    train_sgs = [_mask_labels(subgraphs[b], keep_labels=(b < n_seed))
                 for b in range(n_banks)]

    # ---------------- SILOED ----------------
    # Each bank trains its own GNN purely on its own (label-limited) subgraph.
    silo_w = [gnn.init_weights(in_dim, HIDDEN, seed=seed + 1) for _ in range(n_banks)]
    for b in range(n_banks):
        silo_w[b] = gnn.train_local(silo_w[b], train_sgs[b],
                                    epochs=rounds * local_epochs, lr=lr)
    # Evaluate every siloed model on its bank's TRUE mule labels at the same
    # fixed alert budget used for the federated model.
    siloed_recall = float(np.mean([_recall_at_budget(silo_w[b], subgraphs[b])
                                   for b in range(n_banks)]))
    siloed_auc = float(np.mean([gnn.auc(silo_w[b], subgraphs[b])
                                for b in range(n_banks)]))

    # ---------------- FEDERATED ----------------
    # Shared global GNN; each round: local train -> DP-privatize delta ->
    # robust aggregate -> broadcast. Same loop shape as engine.step().
    global_w = gnn.init_weights(in_dim, HIDDEN, seed=seed + 1)
    for r in range(rounds):
        deltas = []
        for b in range(n_banks):
            wl = gnn.train_local(global_w, train_sgs[b], epochs=local_epochs, lr=lr)
            u = wl - global_w
            u = privatize(u, MAX_NORM, SIGMA, rng)   # DP on the flat delta
            deltas.append(u)
        # Robust aggregation via existing primitive. multi_krum needs n>=4 to
        # tolerate a byzantine; otherwise fall back to fedavg.
        if n_banks >= 4:
            agg, _sel = multi_krum(deltas, n_byzantine=0, m=n_banks)
        else:
            agg = fedavg(deltas)
        global_w = global_w + agg
        if verbose and (r + 1) % max(1, rounds // 5) == 0:
            print(f"  round {r+1:3d}/{rounds}  fed recall="
                  f"{_eval_recall(global_w, subgraphs):.3f}")

    federated_recall = _eval_recall(global_w, subgraphs)
    federated_auc = _eval_auc(global_w, subgraphs)

    return {
        "siloed_recall": siloed_recall,
        "federated_recall": federated_recall,
        "siloed_auc": siloed_auc,
        "federated_auc": federated_auc,
        "lift": federated_recall - siloed_recall,
        "n_banks": n_banks,
        "n_accounts": n_accounts,
        "n_campaigns": n_campaigns,
        "rounds": rounds,
    }


def main():
    print("Federated GNN over cross-bank mule graph")
    print("=" * 52)
    res = run(verbose=True)
    print("-" * 52)
    print(f"banks={res['n_banks']}  accounts={res['n_accounts']}  "
          f"campaigns={res['n_campaigns']}  rounds={res['rounds']}")
    print(f"  SILOED   mule recall: {res['siloed_recall']:.3f}   "
          f"AUC: {res['siloed_auc']:.3f}")
    print(f"  FEDERATED mule recall: {res['federated_recall']:.3f}   "
          f"AUC: {res['federated_auc']:.3f}")
    lift = res["lift"]
    pct = (lift / res["siloed_recall"] * 100) if res["siloed_recall"] > 0 else float("inf")
    print("-" * 52)
    print(f"  LIFT: +{lift:.3f} recall "
          f"({pct:+.1f}% over siloed) from cross-bank federation")
    print("=" * 52)
    return res


if __name__ == "__main__":
    main()
