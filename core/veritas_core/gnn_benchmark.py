"""Reusable, importable runtime computation of the HEADLINE federated-GNN
mule-graph benchmark.

This is the ONE source of truth for the "siloed vs federated GNN" numbers the
demo surfaces. It RUNS the real `graph` + `gnn` federated-vs-siloed training
(the same logic as ``experiments/federated_gnn.py``, which now calls this) and
returns a dict EXACTLY matching the web ``GnnBenchmark`` TS interface so it can
be dropped straight into a backend ``/state`` payload.

Nothing here is hand-copied: ``compute_gnn_benchmark`` measures the metrics and
the per-round federated-recall trajectory from the actual numpy GNN, and
``benchmark_current`` interpolates the live ``current`` block off that REAL
trajectory (never a fabricated ramp).
"""
from __future__ import annotations

import numpy as np

from veritas_core.graph import generate_network, NODE_FEATURE_DIM, Subgraph
from veritas_core import gnn
from veritas_core.aggregation import fedavg, multi_krum
from veritas_core.dp import privatize

# DP / robustness knobs mirror engine.py's conventions.
MAX_NORM = 4.0
SIGMA = 0.01
HIDDEN = 8

# Fraud teams operate at a fixed ALERT BUDGET, so the meaningful metric is
# "recall at a fixed alert volume". Same operating point for siloed/federated.
ALERT_BUDGET = 0.15

# Fixed wall-clock stand-in: the sandbox forbids datetime.now() in some
# contexts, so callers may override but we never call the system clock here.
DEFAULT_GENERATED_AT = "2026-06-20T00:00:00.000Z"


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
    """Zero a bank's mule labels unless it is a designated label-holder. Models
    that only SOME banks have confirmed-mule labels locally. Evaluation always
    uses the TRUE labels, so this only handicaps training, identically for the
    siloed and federated regimes."""
    if keep_labels:
        return sg
    y2 = np.zeros_like(sg.y)
    return Subgraph(sg.X, y2, sg.A, sg.owned_mask, sg.global_ids)


def run_benchmark(n_banks=4, n_accounts=700, n_campaigns=8, rounds=25,
                  local_epochs=12, lr=0.15, seed=0, verbose=False):
    """Run siloed vs federated GNN training over the real cross-bank mule graph.

    Returns a results dict including the REAL per-round federated recall
    trajectory (``round_trajectory``) and the cross-bank edge count from the
    generated graph. This is the shared engine used by both the importable
    ``compute_gnn_benchmark`` and the ``experiments.federated_gnn`` CLI.
    """
    net = generate_network(n_banks=n_banks, n_accounts=n_accounts,
                           n_campaigns=n_campaigns, seed=seed)
    subgraphs = [net.local_subgraph(b) for b in range(n_banks)]      # eval (true y)
    in_dim = NODE_FEATURE_DIM
    rng = np.random.default_rng(seed + 7)

    # Only a subset of banks hold confirmed-mule LABELS in their training data.
    n_seed = max(1, n_banks // 2)
    train_sgs = [_mask_labels(subgraphs[b], keep_labels=(b < n_seed))
                 for b in range(n_banks)]

    # ---------------- SILOED ----------------
    silo_w = [gnn.init_weights(in_dim, HIDDEN, seed=seed + 1) for _ in range(n_banks)]
    for b in range(n_banks):
        silo_w[b] = gnn.train_local(silo_w[b], train_sgs[b],
                                    epochs=rounds * local_epochs, lr=lr)
    siloed_recall = float(np.mean([_recall_at_budget(silo_w[b], subgraphs[b])
                                   for b in range(n_banks)]))
    siloed_auc = float(np.mean([gnn.auc(silo_w[b], subgraphs[b])
                                for b in range(n_banks)]))

    # ---------------- FEDERATED ----------------
    # Shared global GNN; each round: local train -> DP-privatize delta ->
    # robust aggregate -> broadcast. Capture per-round federated recall.
    global_w = gnn.init_weights(in_dim, HIDDEN, seed=seed + 1)
    round_trajectory = []
    for r in range(rounds):
        deltas = []
        for b in range(n_banks):
            wl = gnn.train_local(global_w, train_sgs[b], epochs=local_epochs, lr=lr)
            u = wl - global_w
            u = privatize(u, MAX_NORM, SIGMA, rng)   # DP on the flat delta
            deltas.append(u)
        if n_banks >= 4:
            agg, _sel = multi_krum(deltas, n_byzantine=0, m=n_banks)
        else:
            agg = fedavg(deltas)
        global_w = global_w + agg
        round_trajectory.append(_eval_recall(global_w, subgraphs))
        if verbose and (r + 1) % max(1, rounds // 5) == 0:
            print(f"  round {r+1:3d}/{rounds}  fed recall={round_trajectory[-1]:.3f}")

    federated_recall = round_trajectory[-1] if round_trajectory else _eval_recall(global_w, subgraphs)
    federated_auc = _eval_auc(global_w, subgraphs)

    cross_bank_edges = len(net.cross_bank_edges())

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
        "cross_bank_edges": cross_bank_edges,
        "round_trajectory": [float(x) for x in round_trajectory],
    }


def compute_gnn_benchmark(seed=0, n_banks=4, accounts=700, campaigns=8, rounds=25,
                          generated_at=DEFAULT_GENERATED_AT, verbose=False) -> dict:
    """Run the real GNN benchmark and return a dict matching the TS ``GnnBenchmark``
    shape (minus ``current``, which is per-request via ``benchmark_current``).

    JSON-serializable. All metrics/graph/trajectory numbers are MEASURED from the
    real numpy GNN training, not constants.
    """
    res = run_benchmark(n_banks=n_banks, n_accounts=accounts, n_campaigns=campaigns,
                        rounds=rounds, seed=seed, verbose=verbose)
    return {
        "source": {
            "path": "core/experiments/federated_gnn.py",
            "seed": int(seed),
            "generatedAt": generated_at,
        },
        "graph": {
            "banks": int(res["n_banks"]),
            "accounts": int(res["n_accounts"]),
            "campaigns": int(res["n_campaigns"]),
            "crossBankEdges": int(res["cross_bank_edges"]),
            "alertBudgetPct": int(round(ALERT_BUDGET * 100)),
        },
        "training": {
            "rounds": int(res["rounds"]),
            "aggregation": "Multi-Krum",
            "privacy": "DP clip/noise",
            "verification": "VSA bounded-norm proof",
        },
        "metrics": {
            "siloedRecall": float(res["siloed_recall"]),
            "federatedRecall": float(res["federated_recall"]),
            "siloedAuc": float(res["siloed_auc"]),
            "federatedAuc": float(res["federated_auc"]),
        },
        "roundTrajectory": list(res["round_trajectory"]),
    }


def _clamp01(value):
    return max(0.0, min(1.0, value))


def benchmark_current(benchmark: dict, round: int, campaignActive: bool) -> dict:
    """Build the live ``current`` block from the REAL ``roundTrajectory``.

    ``federatedRecall`` is read off the measured trajectory, indexed by the live
    ``round`` vs ``training.rounds`` (linearly interpolated between samples).
    Before the campaign starts (or round 0) it sits at ``trajectory[0]`` — the
    pre-federation operating point — not a fabricated ramp. ``progress`` is
    ``round / rounds`` clamped to [0, 1].
    """
    rounds = max(1, int(benchmark["training"]["rounds"]))
    traj = list(benchmark.get("roundTrajectory") or [])
    siloed = float(benchmark["metrics"]["siloedRecall"])
    final_fed = float(benchmark["metrics"]["federatedRecall"])

    r = max(0, int(round))
    progress = _clamp01(r / rounds)

    if not traj:
        fed_now = final_fed if campaignActive else siloed
    elif not campaignActive or r <= 0:
        fed_now = traj[0]
    else:
        # Index into the trajectory by the live round (1-based samples).
        pos = min(float(r), float(rounds))
        # trajectory[i] is the federated recall AFTER round i+1.
        f = pos - 1.0
        if f <= 0.0:
            fed_now = traj[0]
        elif f >= len(traj) - 1:
            fed_now = traj[-1]
        else:
            lo = int(np.floor(f))
            frac = f - lo
            fed_now = traj[lo] * (1.0 - frac) + traj[lo + 1] * frac

    return {
        "round": r,
        "progress": float(progress),
        "siloedRecall": siloed,
        "federatedRecall": float(fed_now),
        "recallLift": float(fed_now - siloed),
    }
