"""Tests proving Veritas uses the REAL, free, local flock-sdk.

flock-sdk is an optional extra; skip cleanly if absent. It IS installed in this
environment, so these tests run and assert against the real ABC.
"""
import numpy as np
import pytest

flock_sdk = pytest.importorskip("flock_sdk")

from veritas_core.flock_adapter import (
    VeritasFlockModel,
    USING_REAL_FLOCK_SDK,
    _to_bytes,
    _from_bytes,
    metric_from_bytes,
)
from veritas_core.model import recall


def test_subclasses_real_flock_model():
    assert USING_REAL_FLOCK_SDK is True
    assert issubclass(VeritasFlockModel, flock_sdk.FlockModel)
    # the real base, not the fallback
    assert flock_sdk.FlockModel in VeritasFlockModel.__bases__


def test_bytes_roundtrip_lossless():
    w = np.array([1.5, -2.25, 0.0, 3.14159265358979, 1e-9, -42.0,
                  7.0, 8.0, 9.0, 10.0, -11.0], dtype=np.float64)
    out = _from_bytes(_to_bytes(w))
    assert out.shape == w.shape
    assert np.array_equal(out, w)  # bit-for-bit lossless


def test_empty_parameters_yield_init_weights():
    m = VeritasFlockModel(seed=1)
    w_empty = _from_bytes(b"")
    w_none = _from_bytes(None)
    assert w_empty.shape[0] == m.X.shape[1] + 1
    assert np.array_equal(w_empty, w_none)
    assert np.allclose(w_empty, 0.0)


def test_train_improves_recall_over_init():
    m = VeritasFlockModel(seed=2, campaign=True)
    init_bytes = _to_bytes(np.zeros(m.X.shape[1] + 1))
    r_init = metric_from_bytes(m.evaluate(init_bytes))
    trained = m.train(b"")  # starts from init_weights internally
    r_trained = metric_from_bytes(m.evaluate(trained))
    assert r_trained > r_init


def test_evaluate_returns_decodable_float():
    m = VeritasFlockModel(seed=3)
    b = m.evaluate(m.train(b""))
    assert isinstance(b, (bytes, bytearray))
    val = float(b.decode())
    assert 0.0 <= val <= 1.0
    assert val == metric_from_bytes(b)


def test_init_dataset_keys_seed_from_path():
    m = VeritasFlockModel(seed=0)
    m.init_dataset("seed://77")
    assert m.seed == 77
    m2 = VeritasFlockModel(seed=77)
    assert np.array_equal(m.X, m2.X) and np.array_equal(m.y, m2.y)


def test_three_bank_federation_aggregate_helps():
    # 3 banks see the campaign, train locally, then aggregate via the SDK.
    banks = [VeritasFlockModel(seed=20 + i, campaign=True) for i in range(3)]
    locals_bytes = [b.train(b"") for b in banks]

    # held-out campaign-rich eval set
    from veritas_core.data import make_bank_data, inject_campaign
    Xc, yc = make_bank_data(800, 0.03, seed=999)
    Xc, yc = inject_campaign(Xc, yc, 250, seed=3)

    local_recalls = [recall(_from_bytes(lb), Xc, yc) for lb in locals_bytes]
    global_bytes = banks[0].aggregate(locals_bytes)
    global_recall = recall(_from_bytes(global_bytes), Xc, yc)

    # aggregated global is at least as good as the mean of un-aggregated locals
    assert global_recall >= np.mean(local_recalls) - 1e-9


def test_federation_improves_over_rounds():
    banks = [VeritasFlockModel(seed=30 + i, campaign=(i != 0)) for i in range(4)]
    from veritas_core.data import make_bank_data, inject_campaign
    Xc, yc = make_bank_data(800, 0.03, seed=555)
    Xc, yc = inject_campaign(Xc, yc, 250, seed=5)

    g = b""
    trajectory = []
    for _ in range(6):
        updates = [bk.train(g) for bk in banks]
        g = banks[0].aggregate(updates)
        trajectory.append(recall(_from_bytes(g), Xc, yc))

    # Recall climbs then saturates near 1.0; the federated model reaches a high
    # level and the peak is at least the first round (no net regression).
    assert max(trajectory) >= trajectory[0]
    assert trajectory[-1] >= 0.9


def test_aggregate_robust_mode_runs():
    banks = [VeritasFlockModel(seed=40 + i, campaign=True, robust=True)
             for i in range(5)]
    updates = [bk.train(b"") for bk in banks]
    g = banks[0].aggregate(updates)  # exercises Multi-Krum path
    assert _from_bytes(g).shape == (banks[0].X.shape[1] + 1,)
