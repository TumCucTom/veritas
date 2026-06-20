"""Tests for the LIVE-LOOP ensemble adapter (single-flat-vector interface)."""
import numpy as np
import pytest

from veritas_core import live_ensemble as le
from veritas_core import model as logistic
from veritas_core.aggregation import fedavg, multi_krum
from veritas_core import dp


def test_packed_dim_consistent_across_banks():
    w1 = le.init_weights(seed=0)
    w2 = le.init_weights(seed=1)
    assert w1.ndim == 1
    assert w1.shape == w2.shape
    assert w1.shape[0] == le.weight_dim()


def test_aggregators_accept_packed_vector():
    w1 = le.init_weights(seed=0)
    w2 = le.init_weights(seed=1)
    w3 = le.init_weights(seed=2)

    avg = fedavg([w1, w2])
    assert avg.shape == (le.weight_dim(),)

    agg, sel = multi_krum([w1, w2, w3], n_byzantine=1, m=2)
    assert agg.shape == (le.weight_dim(),)
    assert len(sel) == 2


def test_block_slices_contiguous_and_cover_weight_dim():
    sl = le.block_slices()
    # ordered by start
    ordered = sorted(sl.values(), key=lambda ab: ab[0])
    assert ordered[0][0] == 0
    for (a0, b0), (a1, b1) in zip(ordered, ordered[1:]):
        assert b0 == a1            # contiguous, no gap / overlap
    assert ordered[-1][1] == le.weight_dim()
    # non-overlapping total coverage
    covered = sum(b - a for a, b in sl.values())
    assert covered == le.weight_dim()


def test_predict_proba_and_predict_one_in_range():
    data, y = le.make_live_data(400, seed=3)
    w = le.init_weights(seed=0)
    w = le.train_local(w, data, y, epochs=8)

    p = le.predict_proba(w, data)
    assert p.shape == (len(y),)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)

    # predict_one from a dict of named features
    one = le.predict_one(w, {"amount": 2.0, "velocity": 1.5, "fanout": 1.5})
    assert 0.0 <= one <= 1.0

    # predict_one robust to a bare FEATURE_DIM array
    from veritas_core.data import FEATURE_DIM
    one2 = le.predict_one(w, np.zeros(FEATURE_DIM))
    assert 0.0 <= one2 <= 1.0


def test_train_local_improves_recall():
    data, y = le.make_live_data(600, seed=5)
    w0 = le.init_weights(seed=0)
    r0 = le.recall(w0, data, y)
    w1 = le.train_local(w0, data, y, epochs=12)
    r1 = le.recall(w1, data, y)
    assert r1 > r0


def test_ensemble_beats_logistic_on_heterogeneous_fraud():
    data, y = le.make_live_data(1200, seed=7)

    # train the live ensemble
    we = le.init_weights(seed=0)
    for _ in range(3):
        we = le.train_local(we, data, y, epochs=12)
    rec_ens = le.recall(we, data, y)

    # train a plain logistic model (model.py) on the same tabular view
    from veritas_core.data import FEATURE_DIM
    wl = logistic.init_weights(FEATURE_DIM)
    for _ in range(3):
        wl = logistic.train_local(wl, data.X, y, epochs=12, lr=0.3)
    rec_log = logistic.recall(wl, data.X, y)

    assert rec_ens > rec_log


def test_per_block_dp_clip_keeps_dim():
    data, y = le.make_live_data(300, seed=9)
    w0 = le.init_weights(seed=0)
    w1 = le.train_local(w0, data, y, epochs=6)
    delta = w1 - w0

    clipped = le.clip_blocks(delta, max_norms=1.0)
    assert clipped.shape == delta.shape == (le.weight_dim(),)

    # each block respects its own norm budget after clipping
    for name, (a, b) in le.block_slices().items():
        assert np.linalg.norm(clipped[a:b]) <= 1.0 + 1e-6

    # per-block dict budgets also keep dim
    budgets = {n: 0.5 for n in le.block_slices()}
    clipped2 = le.clip_blocks(delta, max_norms=budgets)
    assert clipped2.shape == (le.weight_dim(),)

    # repack of a privatized full delta keeps dim (global dp helper still works)
    rng = np.random.default_rng(0)
    priv = dp.privatize(delta, max_norm=2.0, sigma=0.1, rng=rng)
    assert priv.shape == (le.weight_dim(),)
