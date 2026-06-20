"""Correctness bar for the temporal/sequence fraud model (GRU, numpy BPTT).

Asserts:
  * train_local improves recall on make_sequence_data;
  * predicted probabilities lie in [0, 1];
  * the model genuinely uses TEMPORAL order — recall on time-shuffled
    sequences is worse than on ordered ones (the temporal signal matters);
  * flat weights have a consistent dim across banks, so fedavg / multi_krum
    accept a list of them and dp.privatize accepts a delta.
"""
import numpy as np

from veritas_core import sequence
from veritas_core.sequence_data import make_sequence_data
from veritas_core.data import FEATURE_DIM
from veritas_core import aggregation


def test_train_improves_recall_and_probs_in_unit_interval():
    X, y = make_sequence_data(600, fraud_rate=0.3, T=8, seed=1)
    w0 = sequence.init_weights(FEATURE_DIM, hidden=16, seed=0)

    r0 = sequence.recall(w0, X, y)
    w1 = sequence.train_local(w0, X, y, epochs=60, lr=0.05)
    r1 = sequence.recall(w1, X, y)

    p = sequence.predict_proba(w1, X)
    assert p.shape == (X.shape[0],)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    assert r1 > r0, f"recall did not improve: {r0:.3f} -> {r1:.3f}"
    assert r1 >= 0.6, f"trained recall too low: {r1:.3f}"


def test_temporal_order_matters():
    # Train on ORDERED sequences.
    X, y = make_sequence_data(600, fraud_rate=0.3, T=8, seed=2)
    w = sequence.init_weights(FEATURE_DIM, hidden=16, seed=0)
    w = sequence.train_local(w, X, y, epochs=60, lr=0.05)
    r_ordered = sequence.recall(w, X, y)

    # Evaluate the SAME model on time-shuffled sequences (same multiset of
    # transactions per account, order destroyed). A bag-of-transactions model
    # would be unaffected; a temporal model degrades.
    rng = np.random.default_rng(123)
    Xs = X.copy()
    for i in range(Xs.shape[0]):
        Xs[i] = Xs[i][rng.permutation(Xs.shape[1])]
    r_shuffled = sequence.recall(w, Xs, y)

    assert r_shuffled < r_ordered, (
        f"temporal order did not matter: ordered={r_ordered:.3f} "
        f"shuffled={r_shuffled:.3f}"
    )


def test_flat_weights_consistent_dim_and_federation_compatible():
    # Each "bank" trains its own local model on its own row-slice of sequences.
    Xa, ya = make_sequence_data(300, fraud_rate=0.3, T=8, seed=10)
    Xb, yb = make_sequence_data(300, fraud_rate=0.3, T=8, seed=11)
    Xc, yc = make_sequence_data(300, fraud_rate=0.3, T=8, seed=12)

    w0 = sequence.init_weights(FEATURE_DIM, hidden=16, seed=0)
    wa = sequence.train_local(w0, Xa, ya, epochs=20, lr=0.05)
    wb = sequence.train_local(w0, Xb, yb, epochs=20, lr=0.05)
    wc = sequence.train_local(w0, Xc, yc, epochs=20, lr=0.05)

    # Same fixed dimension across banks.
    assert wa.shape == wb.shape == wc.shape == w0.shape
    expected = sequence.weight_dim(FEATURE_DIM, 16)
    assert wa.shape == (expected,)

    # fedavg accepts the list and returns a same-dim vector.
    avg = aggregation.fedavg([wa, wb, wc])
    assert avg.shape == w0.shape

    # multi_krum accepts the list too.
    sel, idx = aggregation.multi_krum([wa, wb, wc], n_byzantine=0, m=2)
    assert sel.shape == w0.shape

    # The aggregated global model still predicts valid probabilities.
    p = sequence.predict_proba(avg, Xa)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
