"""From-scratch multilayer-perceptron fraud classifier in pure NUMPY with REAL
hand-derived backprop, drop-in compatible with the same federated-learning
machinery as `model.py` (the live logistic regression).

Why this composes with the federation primitives
------------------------------------------------
Exactly like `model.init_weights` returns a single flat vector, `init_weights`
here returns ONE flat `np.ndarray` packing every parameter (all layer weights +
biases). Its length depends only on `(in_dim, hidden)` — NOT on the number of
samples, and it is identical across every bank that uses the same config. So a
*list* of these vectors drops straight into `aggregation.fedavg` /
`aggregation.multi_krum`, and a single delta vector into `dp.privatize`, with no
changes — the aggregators only ever see a fixed-dim 1-D array of floats.

Architecture
------------
    Linear -> ReLU -> Linear -> ReLU -> ... -> Linear -> sigmoid

For the default HIDDEN=(16, 8) on FEATURE_DIM=10 inputs the layer shapes are::

    (10 -> 16) -> ReLU -> (16 -> 8) -> ReLU -> (8 -> 1) -> sigmoid

i.e. two hidden ReLU layers then a 1-logit readout squashed by sigmoid. The
non-linear hidden layers let the MLP carve up feature space (e.g. XOR-like
interactions) that the single-matrix logistic model in `model.py` cannot.

Weight-vector layout
--------------------
Parameters are packed layer-by-layer, weights then bias, in forward order::

    [ W0.ravel(), b0,  W1.ravel(), b1,  ...,  Wk.ravel(), bk ]

where layer i maps `dims[i] -> dims[i+1]` and `dims = (in_dim, *hidden, 1)`.
`_pack` / `_unpack` are exact inverses. `weight_dim` returns the total length.

Training
--------
Full-batch gradient descent on class-reweighted binary cross-entropy. The
positive class is upweighted by (#neg / #pos) — mirroring `model.train_local`'s
`pw` reweighting — so rare fraud rows are not drowned out. Gradients are
hand-derived through every layer (sigmoid+BCE collapses to `(p - y)` at the
logit; ReLU' gates the upstream gradient). L2 weight decay is applied to the
flat vector exactly as in `model.py`.
"""
import numpy as np

FEATURE_DIM = 10
HIDDEN = (16, 8)  # fixed default so the federation weight vector is consistent


# ---------------------------------------------------------------------------
# Flat weight vector pack / unpack
# ---------------------------------------------------------------------------
def _layer_dims(in_dim, hidden):
    """Full list of layer widths: (in_dim, *hidden, 1)."""
    return [in_dim, *hidden, 1]


def weight_dim(in_dim=FEATURE_DIM, hidden=HIDDEN):
    """Total length of the flat parameter vector for this config.

    sum over layers of (fan_in*fan_out + fan_out). Independent of n samples and
    identical across banks for a given (in_dim, hidden)."""
    dims = _layer_dims(in_dim, hidden)
    total = 0
    for a, b in zip(dims[:-1], dims[1:]):
        total += a * b + b
    return total


def _pack(Ws, bs):
    parts = []
    for W, b in zip(Ws, bs):
        parts.append(W.ravel())
        parts.append(b.ravel())
    return np.concatenate(parts)


def _unpack(w, in_dim, hidden):
    dims = _layer_dims(in_dim, hidden)
    Ws, bs = [], []
    i = 0
    for a, b in zip(dims[:-1], dims[1:]):
        W = w[i:i + a * b].reshape(a, b); i += a * b
        bias = w[i:i + b]; i += b
        Ws.append(W); bs.append(bias)
    return Ws, bs


def init_weights(in_dim=FEATURE_DIM, hidden=HIDDEN, seed=0):
    """Flat np.ndarray of all MLP parameters (He/Glorot-ish init).

    Mirrors `model.init_weights`' contract: a single fixed-dim flat vector that
    fedavg / multi_krum / dp can consume unchanged. Dimension is fixed by
    (in_dim, hidden) and identical across banks."""
    rng = np.random.default_rng(seed)
    dims = _layer_dims(in_dim, hidden)
    Ws, bs = [], []
    for a, b in zip(dims[:-1], dims[1:]):
        scale = np.sqrt(2.0 / a)  # He init for ReLU layers
        Ws.append(rng.normal(0.0, scale, (a, b)))
        bs.append(np.zeros(b))
    return _pack(Ws, bs)


# ---------------------------------------------------------------------------
# Forward / inference
# ---------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _forward(w, X, in_dim, hidden):
    """Forward pass; caches activations and ReLU masks for backprop."""
    Ws, bs = _unpack(w, in_dim, hidden)
    acts = [X]            # acts[0] = input
    masks = []            # relu gate per hidden layer
    h = X
    n_layers = len(Ws)
    for li in range(n_layers):
        pre = h @ Ws[li] + bs[li]
        if li < n_layers - 1:          # hidden layer -> ReLU
            mask = pre > 0.0
            h = pre * mask
            masks.append(mask)
            acts.append(h)
        else:                          # readout -> logit
            z = pre.ravel()
    p = _sigmoid(z)
    return {"Ws": Ws, "bs": bs, "acts": acts, "masks": masks, "z": z, "p": p}


def predict_proba(weights, X, in_dim=FEATURE_DIM, hidden=HIDDEN):
    """Per-sample fraud probability in [0, 1] (Linear->ReLU->...->sigmoid)."""
    return _forward(weights, X, in_dim, hidden)["p"]


# ---------------------------------------------------------------------------
# Training (hand-derived gradients, class-reweighted BCE)
# ---------------------------------------------------------------------------
def train_local(weights, X, y, epochs=15, lr=0.1, l2=1e-4,
                in_dim=FEATURE_DIM, hidden=HIDDEN):
    """Full-batch gradient descent on class-reweighted BCE.

    Mirrors `model.train_local`: positive class upweighted by (#neg/#pos), L2
    decay on the flat vector. Returns the updated flat weight vector (same
    dimension)."""
    w = weights.copy()
    y = y.astype(np.float64)
    n = max(1, len(y))

    npos = int((y == 1).sum())
    pw_pos = ((y == 0).sum() / max(1, npos)) if npos > 0 else 1.0
    sample_w = np.where(y == 1, pw_pos, 1.0)  # (n,)

    for _ in range(epochs):
        f = _forward(w, X, in_dim, hidden)
        Ws, bs, acts, masks, p = f["Ws"], f["bs"], f["acts"], f["masks"], f["p"]

        # dL/dz for weighted BCE with sigmoid: sample_w * (p - y) / n
        dz = (sample_w * (p - y) / n).reshape(-1, 1)   # (n, 1)

        gWs = [None] * len(Ws)
        gbs = [None] * len(bs)

        # readout layer (last)
        last = len(Ws) - 1
        gWs[last] = acts[last].T @ dz                 # (hidden_last, 1)
        gbs[last] = dz.sum(axis=0)                     # (1,)
        delta = dz @ Ws[last].T                        # (n, hidden_last)

        # backprop through hidden layers in reverse
        for li in range(last - 1, -1, -1):
            delta = delta * masks[li]                  # ReLU'
            gWs[li] = acts[li].T @ delta               # (fan_in, fan_out)
            gbs[li] = delta.sum(axis=0)
            delta = delta @ Ws[li].T                   # propagate to prev act

        grad = _pack(gWs, gbs) + l2 * w
        w -= lr * grad

    return w


def recall(weights, X, y, thr=0.5, in_dim=FEATURE_DIM, hidden=HIDDEN):
    """Fraud-detection recall at threshold `thr`."""
    if y.sum() == 0:
        return 1.0
    pred = predict_proba(weights, X, in_dim, hidden) > thr
    return float((pred & (y == 1)).sum() / y.sum())
