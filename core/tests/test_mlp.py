import numpy as np
from veritas_core import mlp, model
from veritas_core.aggregation import fedavg, multi_krum
from veritas_core.data import make_bank_data, FEATURE_DIM


def test_train_improves_recall():
    X, y = make_bank_data(4000, 0.05, seed=3)
    w0 = mlp.init_weights(FEATURE_DIM)
    w1 = mlp.train_local(w0, X, y, epochs=15, lr=0.1)
    assert mlp.recall(w1, X, y) > mlp.recall(w0, X, y) + 0.2


def test_proba_range():
    X, _ = make_bank_data(200, 0.05, seed=4)
    p = mlp.predict_proba(mlp.init_weights(FEATURE_DIM), X)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_weight_dim_fixed_and_consistent():
    # dim independent of n samples, identical across banks for a config
    d = mlp.weight_dim(FEATURE_DIM, (16, 8))
    w1 = mlp.init_weights(FEATURE_DIM, (16, 8), seed=1)
    w2 = mlp.init_weights(FEATURE_DIM, (16, 8), seed=2)
    assert w1.shape == (d,) and w2.shape == (d,)
    # explicit layout check: (10*16+16)+(16*8+8)+(8*1+1)
    assert d == (10 * 16 + 16) + (16 * 8 + 8) + (8 * 1 + 1)


def test_federates_with_existing_primitives():
    # two banks produce same-dim flat vectors -> fedavg / multi_krum accept them
    Xa, ya = make_bank_data(2000, 0.05, seed=10)
    Xb, yb = make_bank_data(2000, 0.05, seed=11)
    w_init = mlp.init_weights(FEATURE_DIM)
    w1 = mlp.train_local(w_init, Xa, ya, epochs=10)
    w2 = mlp.train_local(w_init, Xb, yb, epochs=10)
    assert w1.shape == w2.shape == w_init.shape

    avg = fedavg([w1, w2])
    assert avg.shape == w_init.shape
    # weighted fedavg also works
    avg_w = fedavg([w1, w2], weights=[len(ya), len(yb)])
    assert avg_w.shape == w_init.shape
    # multi_krum over >= 3 same-dim updates
    w3 = mlp.train_local(w_init, Xb, yb, epochs=5)
    krum, sel = multi_krum([w1, w2, w3], n_byzantine=0, m=2)
    assert krum.shape == w_init.shape and len(sel) == 2

    # aggregated weights still produce valid probabilities
    p = mlp.predict_proba(avg, Xa)
    assert p.min() >= 0.0 and p.max() <= 1.0


def _make_xor_data(n, seed):
    """Non-linear (XOR-ish) interaction a linear model cannot capture.

    Fraud iff exactly ONE of features 0, 1 is positive (the sign of their
    product is negative). No single linear boundary separates this; it needs
    a non-linear hidden layer. Remaining features are pure noise.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n, FEATURE_DIM))
    # push features 0,1 away from 0 so the XOR signal is crisp
    X[:, 0] = np.where(rng.random(n) < 0.5, -1.0, 1.0) + rng.normal(0, 0.15, n)
    X[:, 1] = np.where(rng.random(n) < 0.5, -1.0, 1.0) + rng.normal(0, 0.15, n)
    y = ((X[:, 0] * X[:, 1]) < 0).astype(np.int64)  # opposite signs => fraud
    return X.astype(np.float64), y


def test_mlp_beats_logistic_on_nonlinear():
    Xtr, ytr = _make_xor_data(4000, seed=7)
    Xte, yte = _make_xor_data(2000, seed=8)

    # logistic regression (the live model.py)
    wl = model.init_weights(FEATURE_DIM)
    wl = model.train_local(wl, Xtr, ytr, epochs=50, lr=0.3)
    log_recall = model.recall(wl, Xte, yte)

    # MLP through the SAME flat-vector interface
    wm = mlp.init_weights(FEATURE_DIM)
    wm = mlp.train_local(wm, Xtr, ytr, epochs=60, lr=0.2)
    mlp_recall = mlp.recall(wm, Xte, yte)

    assert mlp_recall > log_recall
