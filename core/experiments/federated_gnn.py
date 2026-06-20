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

The actual training is factored into ``veritas_core.gnn_benchmark.run_benchmark``
so the numbers printed here are the SAME numbers the backends serve in /state —
there is ONE source of truth, no hand-copied constants.
"""
from veritas_core.gnn_benchmark import run_benchmark


def run(n_banks=4, n_accounts=700, n_campaigns=8, rounds=25,
        local_epochs=12, lr=0.15, seed=0, verbose=True):
    """Run siloed vs federated GNN training. Returns a results dict.

    Thin wrapper over the shared ``gnn_benchmark.run_benchmark`` so the CLI and
    the backends source numbers from one place and can't drift.
    """
    return run_benchmark(n_banks=n_banks, n_accounts=n_accounts,
                         n_campaigns=n_campaigns, rounds=rounds,
                         local_epochs=local_epochs, lr=lr, seed=seed,
                         verbose=verbose)


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
