"""From-scratch GraphSAGE-style Graph Neural Network in pure NUMPY with REAL
backprop, designed to compose with Veritas' existing federated-learning
primitives (fedavg / multi_krum / dp.privatize).

Architecture (faithful, if small, GNN)
--------------------------------------
One message-passing layer + a linear readout, trained end-to-end:

    H  = relu( Â X W1 + b1 )          # neighbour-aggregating hidden layer
    z  = H W2 + b2                     # readout logit per node
    p  = sigmoid(z)                    # fraud probability per node

where Â = D^{-1}(A + I) is the row-normalised adjacency WITH self loops, so
`Â X` is the mean of each node's own features and its neighbours' features —
exactly GraphSAGE mean aggregation. The layer is permutation-equivariant and a
node's output genuinely depends on its neighbourhood (verified in tests): change
a node's neighbours and `Â X` for that node changes, hence its probability.

Weight vector
-------------
All parameters (W1, b1, W2, b2) are flattened into ONE fixed-length np.ndarray
whose dimension depends only on (in_dim, hidden) — NOT on the number of nodes or
which bank produced it. That is what lets a *list* of these vectors be fed
straight into fedavg / multi_krum, and a single delta vector into dp.privatize,
unchanged. Pack/unpack are exact inverses (verified in tests).

Backprop
--------
Gradients are hand-derived for the binary cross-entropy loss with class
balancing (mule nodes are rare). Only OWNED nodes (subgraph.owned_mask)
contribute to the loss; boundary nodes participate in message passing but never
in supervision — modelling the data boundary.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Flat weight vector pack / unpack
# ---------------------------------------------------------------------------
def weight_dim(in_dim, hidden):
    """Total length of the flat parameter vector."""
    return in_dim * hidden + hidden + hidden + 1  # W1 + b1 + W2 + b2


def init_weights(in_dim, hidden=8, seed=0):
    """Return a flat np.ndarray of all GNN parameters (Glorot-ish init).

    Mirrors model.init_weights' contract: a single flat vector that fedavg /
    multi_krum / dp can consume. Dimension is fixed by (in_dim, hidden)."""
    rng = np.random.default_rng(seed)
    s1 = np.sqrt(2.0 / (in_dim + hidden))
    s2 = np.sqrt(2.0 / (hidden + 1))
    W1 = rng.normal(0, s1, (in_dim, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, s2, (hidden, 1))
    b2 = np.zeros(1)
    return _pack(W1, b1, W2, b2)


def _pack(W1, b1, W2, b2):
    return np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])


def _unpack(w, in_dim, hidden):
    i = 0
    W1 = w[i:i + in_dim * hidden].reshape(in_dim, hidden); i += in_dim * hidden
    b1 = w[i:i + hidden]; i += hidden
    W2 = w[i:i + hidden].reshape(hidden, 1); i += hidden
    b2 = w[i:i + 1]
    return W1, b1, W2, b2


def _infer_dims(w, in_dim):
    """Recover hidden from a flat vector length given in_dim.

    len = in_dim*h + h + h + 1  =>  h = (len - 1) / (in_dim + 2)."""
    hidden = (len(w) - 1) // (in_dim + 2)
    return hidden


# ---------------------------------------------------------------------------
# Forward / inference
# ---------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _forward(w, sg, in_dim, hidden):
    """Full forward pass; returns intermediate activations for backprop."""
    Ahat = sg.normalized_adj()          # (n, n)
    X = sg.X                            # (n, in_dim)
    W1, b1, W2, b2 = _unpack(w, in_dim, hidden)

    AX = Ahat @ X                      # mean-aggregated neighbour features
    pre1 = AX @ W1 + b1                # (n, hidden)
    H = np.maximum(pre1, 0.0)          # relu
    z = (H @ W2).ravel() + b2[0]       # (n,)
    p = _sigmoid(z)
    return {"Ahat": Ahat, "X": X, "AX": AX, "pre1": pre1, "H": H, "z": z, "p": p,
            "W1": W1, "b1": b1, "W2": W2, "b2": b2}


def predict_proba(w, sg, in_dim=None):
    """Per-node fraud probability in [0,1] via message passing.

    `sg` is a graph.Subgraph. `in_dim` defaults to the subgraph feature width.
    """
    if in_dim is None:
        in_dim = sg.X.shape[1]
    hidden = _infer_dims(w, in_dim)
    return _forward(w, sg, in_dim, hidden)["p"]


# ---------------------------------------------------------------------------
# Training (hand-derived gradients, class-balanced BCE over OWNED nodes)
# ---------------------------------------------------------------------------
def train_local(w, sg, epochs=40, lr=0.15, l2=1e-4, in_dim=None):
    """Gradient-descent train on a bank's local subgraph.

    Loss = class-weighted binary cross-entropy over OWNED nodes only. Boundary
    nodes still propagate through `Ahat @ X` (message passing across the bank
    cut) but never appear in the supervised loss. Returns the updated flat
    weight vector (same dimension)."""
    if in_dim is None:
        in_dim = sg.X.shape[1]
    hidden = _infer_dims(w, in_dim)
    w = w.copy()

    owned = sg.owned_mask
    y = sg.y.astype(np.float64)
    n_owned = max(1, int(owned.sum()))

    # class balancing: upweight rare mule nodes among OWNED nodes
    npos = int(((y == 1) & owned).sum())
    nneg = int(((y == 0) & owned).sum())
    pw_pos = (nneg / max(1, npos)) if npos > 0 else 1.0
    sample_w = np.where(y == 1, pw_pos, 1.0) * owned  # boundary -> 0 weight
    sw_sum = max(1e-8, sample_w.sum())

    for _ in range(epochs):
        f = _forward(w, sg, in_dim, hidden)
        p, H, AX, Ahat = f["p"], f["H"], f["AX"], f["Ahat"]
        W2 = f["W2"]

        # dL/dz for weighted BCE: sample_w * (p - y) / sum(sample_w)
        dz = (sample_w * (p - y)) / sw_sum          # (n,)

        # readout grads
        gW2 = H.T @ dz.reshape(-1, 1)               # (hidden, 1)
        gb2 = np.array([dz.sum()])

        # backprop into hidden
        dH = np.outer(dz, W2.ravel())               # (n, hidden)
        dpre1 = dH * (f["pre1"] > 0.0)              # relu'
        gW1 = AX.T @ dpre1                          # (in_dim, hidden)
        gb1 = dpre1.sum(axis=0)

        grad = _pack(gW1, gb1, gW2, gb2) + l2 * w
        w -= lr * grad

    return w


def recall(w, sg, thr=0.5, in_dim=None):
    """Mule-detection recall over OWNED mule nodes."""
    if in_dim is None:
        in_dim = sg.X.shape[1]
    p = predict_proba(w, sg, in_dim)
    owned = sg.owned_mask
    y = sg.y
    mask = owned & (y == 1)
    if mask.sum() == 0:
        return 1.0
    pred = p > thr
    return float((pred & mask).sum() / mask.sum())


def auc(w, sg, in_dim=None):
    """ROC-AUC over OWNED nodes (rank-based, ties averaged).

    Delegates to the single canonical tie-averaged Mann-Whitney AUC in
    :func:`veritas_core.ensemble.auc` so there is exactly one correct
    implementation (the previous local copy ranked but did NOT average tied
    ranks, returning 0.5 instead of 0.625 on tied scores)."""
    from .ensemble import auc as _auc
    if in_dim is None:
        in_dim = sg.X.shape[1]
    p = predict_proba(w, sg, in_dim)
    owned = sg.owned_mask
    return _auc(p[owned], sg.y[owned])
