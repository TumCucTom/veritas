"""Tests for the REAL ULB credit-card fraud loader, non-IID partition, and the
federated-vs-siloed lift on a SMALL seeded subsample (kept fast).

If the CSV is absent the data-dependent tests skip gracefully. It IS present in
this environment, so they run.
"""
import os

import numpy as np
import pytest

from veritas_core import realdata
from veritas_core import model as logistic
from veritas_core.aggregation import fedavg

# Resolve the CSV regardless of pytest's cwd (repo-root or core/).
_CANDIDATES = [
    "core/data/creditcard.csv",
    "data/creditcard.csv",
    os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv"),
]
_CSV = next((p for p in _CANDIDATES if os.path.exists(p)), None)

requires_csv = pytest.mark.skipif(_CSV is None, reason="creditcard.csv not present")


def _auprc(scores, y):
    scores = np.asarray(scores, dtype=np.float64); y = np.asarray(y).astype(int)
    P = int((y == 1).sum())
    if P == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys == 1); fp = np.cumsum(ys == 0)
    precision = tp / np.clip(tp + fp, 1, None)
    recall = tp / P
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - rec_prev) * precision))


@pytest.fixture(scope="module")
def loaded():
    if _CSV is None:
        pytest.skip("creditcard.csv not present")
    X, y = realdata.load_creditcard(path=_CSV, download=False, standardize=True)
    return X, y


@requires_csv
def test_loader_shapes_and_labels(loaded):
    X, y = loaded
    # 30 features (V1..V28 + log-amount + normalised time), aligned rows.
    assert X.ndim == 2 and X.shape[1] == 30
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 280_000          # full real dataset
    # y strictly in {0, 1}
    assert set(np.unique(y).tolist()) <= {0, 1}
    assert y.dtype.kind in "iu"


@requires_csv
def test_realistic_fraud_rate(loaded):
    _, y = loaded
    rate = y.mean()
    # ULB set is ~0.173% fraud; allow a generous band.
    assert 0.0010 < rate < 0.0030
    assert int(y.sum()) == 492


@requires_csv
def test_standardization(loaded):
    X, _ = loaded
    # z-scored columns: ~0 mean, ~unit variance.
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(X.std(axis=0), 1.0, atol=1e-6)


@requires_csv
def test_partition_covers_all_rows_once(loaded):
    X, y = loaded
    n_banks = 5
    banks = realdata.partition_noniid(X, y, n_banks, seed=0)
    assert len(banks) == n_banks
    total = sum(len(b[1]) for b in banks)
    assert total == len(y)                       # exact partition: every row once
    assert sum(int(b[1].sum()) for b in banks) == int(y.sum())  # all frauds kept


@requires_csv
def test_partition_is_noniid(loaded):
    X, y = loaded
    banks = realdata.partition_noniid(X, y, 5, seed=0, concentration="skewed")
    rates = np.array([b[1].mean() for b in banks])
    # Non-IID => fraud rates differ markedly across banks.
    assert rates.max() > 2.0 * (rates.min() + 1e-9)
    counts = np.array([int(b[1].sum()) for b in banks])
    assert counts.max() > counts.min()           # skew present


@requires_csv
def test_federated_beats_siloed_auprc():
    """On a small seeded subsample, federated AUPRC >= mean siloed AUPRC."""
    if _CSV is None:
        pytest.skip("creditcard.csv not present")
    rng = np.random.default_rng(0)
    X, y = realdata.load_creditcard(path=_CSV, download=False, standardize=True)

    # Small, fast subsample: keep ALL frauds + a capped slice of negatives.
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    neg = rng.choice(neg, size=12_000, replace=False)
    idx = np.concatenate([pos, neg]); rng.shuffle(idx)
    Xs, ys = X[idx], y[idx]

    # stratified train/test
    p = np.where(ys == 1)[0]; n = np.where(ys == 0)[0]
    rng.shuffle(p); rng.shuffle(n)
    npte = max(1, int(len(p) * 0.3)); nnte = max(1, int(len(n) * 0.3))
    te = np.concatenate([p[:npte], n[:nnte]])
    tr = np.concatenate([p[npte:], n[nnte:]])
    Xtr, ytr, Xte, yte = Xs[tr], ys[tr], Xs[te], ys[te]

    banks = realdata.partition_noniid(Xtr, ytr, 5, seed=0, concentration="skewed")

    # siloed: mean per-bank logistic AUPRC
    sil = []
    for b in banks:
        w = logistic.train_local(logistic.init_weights(b[0].shape[1]),
                                 b[0], b[1], epochs=40, lr=0.3)
        sil.append(_auprc(logistic.predict_proba(w, Xte), yte))
    mean_sil = float(np.mean(sil))

    # federated: fedavg over a few rounds
    w = logistic.init_weights(Xtr.shape[1])
    for _ in range(6):
        updates = [logistic.train_local(w, b[0], b[1], epochs=12, lr=0.3)
                   for b in banks]
        w = fedavg(updates)
    fed = _auprc(logistic.predict_proba(w, Xte), yte)

    assert yte.sum() > 0
    # The lift: federated at least matches the average siloed bank.
    assert fed >= mean_sil - 1e-6
