"""Learned shared-categorical-FEATURE embeddings + MLP-head fraud classifier in
pure NUMPY with REAL hand-derived backprop, drop-in compatible with the same
federated-learning machinery as `model.py` / `mlp.py`.

Why SHARED categorical features (and NOT per-account ids)
--------------------------------------------------------
Federated learning only helps if the *meaning* of an embedded index is the same
at every bank. A few categorical fields satisfy this because they draw from a
small, standardised, cross-institution vocabulary:

    * merchant-category code (e.g. ISO 18245 MCC),
    * counterparty country (ISO 3166),
    * payment channel / rail (FasterPayments, CHAPS, card, ...).

Embedding-row `k` for "country = GB" means GB at bank A and bank B, so averaging
the two banks' embedding tables (FedAvg) is meaningful and the learned scam
geometry transfers. The fraud signal in practice lives in the *interaction* of
these fields (e.g. a specific merchant-category x destination-country combo that
is a known mule corridor) — a linear model on one-hot columns struggles to carve
that out, whereas a dense embedding + non-linear head learns it directly.

Per-account / per-customer ids are the OPPOSITE: account "12345" at bank A and
account "12345" at bank B are unrelated entities, so their embedding rows do not
align across banks. Averaging them is nonsense and would leak / corrupt. They are
also unbounded-cardinality and privacy-sensitive. We therefore embed ONLY shared
low-cardinality categorical fields and leave identity to per-bank local features.

Architecture
------------
    [emb(cat_0) | emb(cat_1) | ... | dense_features]  ->  Linear -> ReLU
                                                      ->  Linear -> sigmoid

i.e. each categorical field has its own embedding table (cardinality_i x emb_dim);
the looked-up rows are concatenated with the raw dense feature vector and fed to a
small one-hidden-layer ReLU MLP that outputs one fraud logit.

Federation: fixed-dim flat weight vector
----------------------------------------
Exactly like `model.init_weights` / `mlp.init_weights`, `init_weights` returns ONE
flat `np.ndarray` packing every parameter (all embedding tables + the head's
weights and biases). Its length depends ONLY on `(cardinalities, emb_dim,
dense_dim, hidden)` — NOT on the data — and is identical across every bank that
shares the same categorical schema. So a *list* of these vectors drops straight
into `aggregation.fedavg` / `aggregation.multi_krum`, and a single delta into
`dp.privatize`, with no changes: the aggregators only ever see a fixed-dim 1-D
array of floats. This is exactly why we embed SHARED categoricals — only then is
row `k` of every bank's table the same concept, so averaging the flat vector is
sound.

Weight-vector layout
--------------------
Packed in a fixed order, flattened then concatenated::

    [ E_0.ravel(), E_1.ravel(), ..., E_{F-1}.ravel(),
      W0.ravel(), b0, W1.ravel(), b1 ]

where `E_i` is the (cardinality_i, emb_dim) table for categorical field i, and
`(W0,b0)`/`(W1,b1)` are the hidden / readout layers of the head whose input width
is `F*emb_dim + dense_dim`. `_pack` / `_unpack` are exact inverses; `weight_dim`
returns the total length.

Training
--------
Full-batch gradient descent on class-reweighted binary cross-entropy (positive
class upweighted by #neg/#pos, mirroring `model.train_local`). Gradients are
hand-derived through every layer; the embedding-table gradient is the scatter-add
of the input-layer gradient back onto the rows that were looked up. L2 weight
decay is applied to the flat vector exactly as in `model.py`.
"""
import numpy as np

from .data import FEATURE_DIM


# ---------------------------------------------------------------------------
# Flat weight vector pack / unpack
# ---------------------------------------------------------------------------
def _head_dims(in_dim, hidden):
    """Head layer widths: (in_dim, hidden, 1)."""
    return [in_dim, hidden, 1]


def weight_dim(cardinalities, emb_dim=4, dense_dim=FEATURE_DIM, hidden=16):
    """Total length of the flat parameter vector for this config.

    sum_i(cardinality_i*emb_dim)  +  head params. Independent of n samples and
    identical across banks that share the same categorical schema."""
    total = sum(c * emb_dim for c in cardinalities)
    head_in = len(cardinalities) * emb_dim + dense_dim
    dims = _head_dims(head_in, hidden)
    for a, b in zip(dims[:-1], dims[1:]):
        total += a * b + b
    return total


def _pack(Es, Ws, bs):
    parts = [E.ravel() for E in Es]
    for W, b in zip(Ws, bs):
        parts.append(W.ravel())
        parts.append(b.ravel())
    return np.concatenate(parts)


def _unpack(w, cardinalities, emb_dim, dense_dim, hidden):
    Es = []
    i = 0
    for c in cardinalities:
        Es.append(w[i:i + c * emb_dim].reshape(c, emb_dim))
        i += c * emb_dim
    head_in = len(cardinalities) * emb_dim + dense_dim
    dims = _head_dims(head_in, hidden)
    Ws, bs = [], []
    for a, b in zip(dims[:-1], dims[1:]):
        W = w[i:i + a * b].reshape(a, b); i += a * b
        bias = w[i:i + b]; i += b
        Ws.append(W); bs.append(bias)
    return Es, Ws, bs


def init_weights(cardinalities, emb_dim=4, dense_dim=FEATURE_DIM, hidden=16,
                 seed=0):
    """Flat np.ndarray of all parameters (embedding tables + MLP head).

    Mirrors `model.init_weights`' contract: a single fixed-dim flat vector that
    fedavg / multi_krum / dp consume unchanged. Dimension is fixed by
    `(cardinalities, emb_dim, dense_dim, hidden)` and identical across banks that
    share the categorical schema.

    Parameters
    ----------
    cardinalities : list[int]
        Number of distinct categories for each SHARED categorical field, in a
        fixed schema order agreed across banks (e.g. [n_merchant_cat, n_country,
        n_channel]).
    """
    rng = np.random.default_rng(seed)
    Es = [rng.normal(0.0, 0.1, (c, emb_dim)) for c in cardinalities]
    head_in = len(cardinalities) * emb_dim + dense_dim
    dims = _head_dims(head_in, hidden)
    Ws, bs = [], []
    for a, b in zip(dims[:-1], dims[1:]):
        scale = np.sqrt(2.0 / a)  # He init for the ReLU head
        Ws.append(rng.normal(0.0, scale, (a, b)))
        bs.append(np.zeros(b))
    return _pack(Es, Ws, bs)


# ---------------------------------------------------------------------------
# Forward / inference
# ---------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _embed_lookup(Es, X_cat):
    """Concatenate looked-up embedding rows across all categorical fields.

    X_cat : (n, F) int array of category indices. Returns (n, F*emb_dim)."""
    cols = [E[X_cat[:, j]] for j, E in enumerate(Es)]  # each (n, emb_dim)
    return np.concatenate(cols, axis=1)


def _forward(w, X_cat, X_dense, cardinalities, emb_dim, dense_dim, hidden):
    """Forward pass; caches activations / ReLU mask for backprop."""
    Es, Ws, bs = _unpack(w, cardinalities, emb_dim, dense_dim, hidden)
    emb = _embed_lookup(Es, X_cat)             # (n, F*emb_dim)
    a0 = np.concatenate([emb, X_dense], axis=1)  # (n, head_in)
    pre0 = a0 @ Ws[0] + bs[0]
    mask = pre0 > 0.0
    a1 = pre0 * mask                            # ReLU
    z = (a1 @ Ws[1] + bs[1]).ravel()
    p = _sigmoid(z)
    return {"Es": Es, "Ws": Ws, "bs": bs, "emb": emb, "a0": a0,
            "mask": mask, "a1": a1, "z": z, "p": p}


def predict_proba(weights, X_cat, X_dense, cardinalities, emb_dim=4,
                  dense_dim=FEATURE_DIM, hidden=16):
    """Per-sample fraud probability in [0, 1].

    Looks up each categorical field's embedding, concatenates with the dense
    feature vector, and runs the ReLU-MLP head -> sigmoid."""
    X_cat = np.asarray(X_cat)
    return _forward(weights, X_cat, X_dense, cardinalities, emb_dim,
                    dense_dim, hidden)["p"]


# ---------------------------------------------------------------------------
# Training (hand-derived gradients incl. embedding-table updates)
# ---------------------------------------------------------------------------
def train_local(weights, X_cat, X_dense, y, epochs=20, lr=0.1, l2=1e-4,
                cardinalities=None, emb_dim=4, dense_dim=FEATURE_DIM, hidden=16):
    """Full-batch gradient descent on class-reweighted BCE.

    Mirrors `model.train_local`: positive class upweighted by (#neg/#pos), L2
    decay on the flat vector. The embedding-table gradient is the scatter-add of
    the head's input-layer gradient onto the rows that were looked up. Returns
    the updated flat weight vector (same dimension)."""
    if cardinalities is None:
        raise ValueError("cardinalities is required (the categorical schema)")
    X_cat = np.asarray(X_cat)
    w = weights.copy()
    y = y.astype(np.float64)
    n = max(1, len(y))
    F = len(cardinalities)

    npos = int((y == 1).sum())
    pw_pos = ((y == 0).sum() / max(1, npos)) if npos > 0 else 1.0
    sample_w = np.where(y == 1, pw_pos, 1.0)

    for _ in range(epochs):
        f = _forward(w, X_cat, X_dense, cardinalities, emb_dim, dense_dim,
                     hidden)
        Es, Ws, bs = f["Es"], f["Ws"], f["bs"]
        a0, mask, a1, p = f["a0"], f["mask"], f["a1"], f["p"]

        # dL/dz for weighted BCE with sigmoid: sample_w * (p - y) / n
        dz = (sample_w * (p - y) / n).reshape(-1, 1)         # (n, 1)

        # readout layer
        gW1 = a1.T @ dz                                      # (hidden, 1)
        gb1 = dz.sum(axis=0)                                 # (1,)
        d1 = dz @ Ws[1].T                                    # (n, hidden)

        # hidden layer (ReLU')
        d1 = d1 * mask
        gW0 = a0.T @ d1                                      # (head_in, hidden)
        gb0 = d1.sum(axis=0)                                 # (hidden,)
        d_in = d1 @ Ws[0].T                                  # (n, head_in)

        # gradient flowing into the concatenated embedding block (first F*emb_dim
        # cols of the head input); scatter-add back onto the embedding tables.
        d_emb = d_in[:, :F * emb_dim]                        # (n, F*emb_dim)
        gEs = []
        for j, c in enumerate(cardinalities):
            sl = d_emb[:, j * emb_dim:(j + 1) * emb_dim]     # (n, emb_dim)
            gE = np.zeros((c, emb_dim))
            np.add.at(gE, X_cat[:, j], sl)                   # scatter-add
            gEs.append(gE)

        grad = _pack(gEs, [gW0, gW1], [gb0, gb1]) + l2 * w
        w -= lr * grad

    return w


def recall(weights, X_cat, X_dense, y, thr=0.5, cardinalities=None, emb_dim=4,
           dense_dim=FEATURE_DIM, hidden=16):
    """Fraud-detection recall at threshold `thr`."""
    if cardinalities is None:
        raise ValueError("cardinalities is required (the categorical schema)")
    if y.sum() == 0:
        return 1.0
    p = predict_proba(weights, X_cat, X_dense, cardinalities, emb_dim,
                      dense_dim, hidden)
    pred = p > thr
    return float((pred & (y == 1)).sum() / y.sum())


# ---------------------------------------------------------------------------
# Synthetic data with a fraudulent CATEGORICAL INTERACTION
# ---------------------------------------------------------------------------
def make_categorical_data(n, fraud_rate=0.06, n_merchant_cat=12, n_country=8,
                          n_channel=4, dense_dim=FEATURE_DIM, seed=0):
    """Synthetic data where a specific (merchant-category x country) COMBO is
    highly fraudulent — an interaction a dense-only model cannot see at all but
    embeddings can.

    The "mule corridor" is `merchant_cat == 3 AND country == 5`: neither value is
    fraudulent on its own (each appears plenty in legit rows), only their
    co-occurrence is. We deliberately make the corridor a LARGE share of all
    fraud and give the dense features only a *weak, overlapping* bump for the
    remaining background fraud. So a dense-only baseline can never recover the
    corridor positives (no dense signal there), while the embedding model reads
    the categorical interaction directly and wins on recall.

    Returns
    -------
    X_cat : (n, 3) int64   columns = [merchant_cat, country, channel]
    X_dense : (n, dense_dim) float64
    y : (n,) int64
    cardinalities : list[int]  -> [n_merchant_cat, n_country, n_channel]
    """
    rng = np.random.default_rng(seed)
    cardinalities = [n_merchant_cat, n_country, n_channel]

    merchant = rng.integers(0, n_merchant_cat, n)
    country = rng.integers(0, n_country, n)
    channel = rng.integers(0, n_channel, n)

    # Force the corridor combo to be common enough to dominate the fraud set:
    # stamp it onto a chunk of rows so corridor fraud is the majority typology.
    corridor_seed = rng.random(n) < 0.10
    merchant[corridor_seed] = 3
    country[corridor_seed] = 5

    corridor = (merchant == 3) & (country == 5)
    corridor_fraud = corridor & (rng.random(n) < 0.85)

    # Background fraud: a smaller, weaker dense-only typology.
    base = (rng.random(n) < fraud_rate)
    y = (base | corridor_fraud).astype(np.int64)

    # Dense features carry ONLY a weak bump for background fraud, NOTHING for the
    # corridor fraud — the corridor signal is reachable solely via the
    # categorical interaction, not the dense vector. The bump is modest and
    # noisy so the dense baseline cannot lean on it to recover corridor rows.
    X_dense = rng.normal(0.0, 1.0, (n, dense_dim))
    bg_only = base & ~corridor_fraud
    X_dense[bg_only, 0] += 0.8
    if dense_dim > 6:
        X_dense[bg_only, 6] += 0.8

    X_cat = np.stack([merchant, country, channel], axis=1).astype(np.int64)
    return X_cat, X_dense.astype(np.float64), y, cardinalities
