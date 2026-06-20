from experiments.federated_gnn import run


def test_federated_beats_or_matches_siloed():
    res = run(n_banks=4, n_accounts=700, n_campaigns=8, rounds=25,
              local_epochs=12, lr=0.15, seed=0, verbose=False)
    # The cross-bank mule topology recurs across banks; pooling structural
    # signal via federation must not underperform isolated siloed training.
    assert res["federated_recall"] >= res["siloed_recall"]


def test_federated_detects_mules():
    res = run(seed=0, verbose=False)
    # federation should achieve meaningful recall (not a degenerate pass)
    assert res["federated_recall"] >= 0.5


def test_deterministic():
    a = run(seed=3, verbose=False)
    b = run(seed=3, verbose=False)
    assert a["federated_recall"] == b["federated_recall"]
    assert a["siloed_recall"] == b["siloed_recall"]
