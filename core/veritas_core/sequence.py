"""From-scratch GRU sequence model in pure NUMPY with hand-derived BPTT, for
TEMPORAL fraud (mule layering / velocity) — a sibling of the tabular `model.py`
and graph `gnn.py` that federates through the SAME machinery.

Why a recurrent model
---------------------
Fraud is temporal: an account's transactions over time carry a signature (a
quiet period then a high-value high-fan-out burst — layering) that a
per-transaction tabular model cannot see. We run a GRU over each account's
sequence of T transaction feature vectors and read out the FINAL hidden state
through a sigmoid to a fraud probability. Because the GRU consumes the sequence
in time order, the output depends on the *order* of the timesteps (verified in
tests: shuffling the timestep order degrades recall).

Architecture
------------
Single-layer GRU, hidden size H, over inputs x_t in R^feat:

    z_t = sigmoid(x_t Wz + h_{t-1} Uz + bz)          # update gate
    r_t = sigmoid(x_t Wr + h_{t-1} Ur + br)          # reset gate
    n_t = tanh  (x_t Wn + (r_t * h_{t-1}) Un + bn)   # candidate
    h_t = (1 - z_t) * h_{t-1} + z_t * n_t            # new hidden

After T steps, a linear readout on the final hidden state:

    logit = h_T . Wo + bo
    p     = sigmoid(logit)                            # fraud prob per sequence

Flat weight vector
------------------
Every parameter (Wz,Uz,bz, Wr,Ur,br, Wn,Un,bn, Wo,bo) is flattened into ONE
fixed-length np.ndarray whose dimension depends ONLY on (feat, hidden) — never
on n, T, or which bank produced it. That fixed dim is what lets a *list* of
these vectors go straight into aggregation.fedavg / aggregation.multi_krum, and
a single delta into dp.privatize, exactly like model.py / gnn.py.

BPTT
----
Gradients are hand-derived for class-reweighted binary cross-entropy on the
readout, backpropagated through time across all GRU gates (no autograd). Forward
caches per-timestep gates/hiddens; the backward pass walks t = T-1 .. 0 carrying
dh into h_{t-1}.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Flat weight vector pack / unpack
#   blocks, in order:
#     Wz (feat,H) Uz (H,H) bz (H,)
#     Wr (feat,H) Ur (H,H) br (H,)
#     Wn (feat,H) Un (H,H) bn (H,)
#     Wo (H,1)    bo (1,)
# ---------------------------------------------------------------------------
def weight_dim(feat, hidden):
    """Total length of the flat parameter vector for given (feat, hidden)."""
    gate = feat * hidden + hidden * hidden + hidden  # W + U + b per gate
    return 3 * gate + hidden + 1                     # 3 gates + Wo + bo


def init_weights(feat=None, hidden=16, seed=0):
    """Return a flat np.ndarray of all GRU parameters (Glorot/orthogonal-ish).

    `feat` defaults to data.FEATURE_DIM. Dimension is fixed by (feat, hidden),
    making the vector federation-ready: fedavg / multi_krum accept a list of
    these as long as every bank uses the same (feat, hidden)."""
    if feat is None:
        from .data import FEATURE_DIM
        feat = FEATURE_DIM
    rng = np.random.default_rng(seed)
    sx = np.sqrt(1.0 / feat)
    sh = np.sqrt(1.0 / hidden)
    so = np.sqrt(1.0 / hidden)

    def W():
        return rng.normal(0.0, sx, (feat, hidden))

    def U():
        return rng.normal(0.0, sh, (hidden, hidden))

    parts = [
        W(), U(), np.zeros(hidden),   # z gate
        W(), U(), np.zeros(hidden),   # r gate
        W(), U(), np.zeros(hidden),   # n gate
        rng.normal(0.0, so, (hidden, 1)), np.zeros(1),  # readout
    ]
    return np.concatenate([p.ravel() for p in parts])


def _unpack(w, feat, hidden):
    i = 0
    fh, hh = feat * hidden, hidden * hidden

    def take(k, shape):
        nonlocal i
        out = w[i:i + k].reshape(shape)
        i += k
        return out

    Wz = take(fh, (feat, hidden)); Uz = take(hh, (hidden, hidden)); bz = take(hidden, (hidden,))
    Wr = take(fh, (feat, hidden)); Ur = take(hh, (hidden, hidden)); br = take(hidden, (hidden,))
    Wn = take(fh, (feat, hidden)); Un = take(hh, (hidden, hidden)); bn = take(hidden, (hidden,))
    Wo = take(hidden, (hidden, 1)); bo = take(1, (1,))
    return Wz, Uz, bz, Wr, Ur, br, Wn, Un, bn, Wo, bo


def _pack_grads(g):
    return np.concatenate([p.ravel() for p in g])


def _infer_hidden(w, feat):
    """Recover hidden from flat length L and feat.

    L = 3*(feat*H + H^2 + H) + H + 1
      = 3*H^2 + (3*feat + 4)*H + 1
    Solve the quadratic 3 H^2 + (3 feat + 4) H + (1 - L) = 0."""
    L = len(w)
    a = 3.0
    b = 3.0 * feat + 4.0
    c = 1.0 - L
    disc = b * b - 4 * a * c
    H = (-b + np.sqrt(disc)) / (2 * a)
    return int(round(H))


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _forward(w, X, feat, hidden, cache=False):
    """Run the GRU over X (n, T, feat). Returns p (n,) and optional cache.

    Vectorised over the batch n; loops over the T timesteps."""
    Wz, Uz, bz, Wr, Ur, br, Wn, Un, bn, Wo, bo = _unpack(w, feat, hidden)
    n, T, _ = X.shape
    h = np.zeros((n, hidden))

    hs = [h]               # h_0 .. h_T
    zs, rs, ns, hprevs = [], [], [], []
    for t in range(T):
        xt = X[:, t, :]                              # (n, feat)
        z = _sigmoid(xt @ Wz + h @ Uz + bz)          # (n, H)
        r = _sigmoid(xt @ Wr + h @ Ur + br)
        nc = np.tanh(xt @ Wn + (r * h) @ Un + bn)
        h_new = (1.0 - z) * h + z * nc
        if cache:
            zs.append(z); rs.append(r); ns.append(nc); hprevs.append(h)
        h = h_new
        hs.append(h)

    logit = (h @ Wo).ravel() + bo[0]                 # final hidden readout
    p = _sigmoid(logit)
    if not cache:
        return p, None
    c = {
        "X": X, "zs": zs, "rs": rs, "ns": ns, "hprevs": hprevs, "hs": hs,
        "hT": h, "p": p,
        "Wz": Wz, "Uz": Uz, "Wr": Wr, "Ur": Ur, "Wn": Wn, "Un": Un, "Wo": Wo,
    }
    return p, c


def predict_proba(weights, X, feat=None):
    """Fraud probability per sequence in [0,1]. X is (n, T, feat)."""
    if feat is None:
        feat = X.shape[2]
    hidden = _infer_hidden(weights, feat)
    p, _ = _forward(weights, X, feat, hidden, cache=False)
    return p


# ---------------------------------------------------------------------------
# Training (hand-derived BPTT, class-reweighted BCE)
# ---------------------------------------------------------------------------
def train_local(weights, X, y, epochs=60, lr=0.05, l2=1e-5, feat=None):
    """BPTT-train the GRU on a bank's local sequences.

    Loss = class-weighted binary cross-entropy on the readout (mule sequences
    are rare -> upweighted). Returns an updated flat weight vector of the SAME
    dimension, ready for fedavg / multi_krum / dp.privatize."""
    if feat is None:
        feat = X.shape[2]
    hidden = _infer_hidden(weights, feat)
    w = weights.copy()
    n, T, _ = X.shape
    y = y.astype(np.float64)

    npos = int((y == 1).sum())
    nneg = int((y == 0).sum())
    pw_pos = (nneg / max(1, npos)) if npos > 0 else 1.0
    sample_w = np.where(y == 1, pw_pos, 1.0)
    sw_sum = max(1e-8, sample_w.sum())

    for _ in range(epochs):
        p, c = _forward(w, X, feat, hidden, cache=True)
        Wz, Uz = c["Wz"], c["Uz"]
        Wr, Ur = c["Wr"], c["Ur"]
        Wn, Un = c["Wn"], c["Un"]
        Wo = c["Wo"]

        # dL/dlogit for weighted BCE
        dlogit = (sample_w * (p - y)) / sw_sum        # (n,)

        # readout grads
        gWo = c["hT"].T @ dlogit.reshape(-1, 1)       # (H,1)
        gbo = np.array([dlogit.sum()])

        # gradient flowing into final hidden state
        dh = dlogit.reshape(-1, 1) * Wo.ravel()[None, :]   # (n,H)

        gWz = np.zeros_like(Wz); gUz = np.zeros_like(Uz); gbz = np.zeros(hidden)
        gWr = np.zeros_like(Wr); gUr = np.zeros_like(Ur); gbr = np.zeros(hidden)
        gWn = np.zeros_like(Wn); gUn = np.zeros_like(Un); gbn = np.zeros(hidden)

        for t in range(T - 1, -1, -1):
            xt = c["X"][:, t, :]
            z = c["zs"][t]; r = c["rs"][t]; nc = c["ns"][t]; hprev = c["hprevs"][t]

            # h_t = (1-z)*hprev + z*nc
            dz = dh * (nc - hprev)                    # through gate z
            dnc = dh * z                              # through candidate
            dh_prev = dh * (1.0 - z)                  # direct skip path

            # candidate: nc = tanh(a_n), a_n = xt Wn + (r*hprev) Un + bn
            da_n = dnc * (1.0 - nc * nc)              # tanh'
            gWn += xt.T @ da_n
            rh = r * hprev
            gUn += rh.T @ da_n
            gbn += da_n.sum(axis=0)
            drh = da_n @ Un.T                          # (n,H) into (r*hprev)
            dr = drh * hprev
            dh_prev += drh * r

            # reset gate: r = sigmoid(a_r)
            da_r = dr * r * (1.0 - r)
            gWr += xt.T @ da_r
            gUr += hprev.T @ da_r
            gbr += da_r.sum(axis=0)
            dh_prev += da_r @ Ur.T

            # update gate: z = sigmoid(a_z)
            da_z = dz * z * (1.0 - z)
            gWz += xt.T @ da_z
            gUz += hprev.T @ da_z
            gbz += da_z.sum(axis=0)
            dh_prev += da_z @ Uz.T

            dh = dh_prev                               # carry to previous step

        grad = _pack_grads([gWz, gUz, gbz, gWr, gUr, gbr, gWn, gUn, gbn, gWo, gbo])
        grad += l2 * w
        w -= lr * grad

    return w


def recall(weights, X, y, thr=0.5, feat=None):
    """Mule-detection recall over fraud sequences."""
    if y.sum() == 0:
        return 1.0
    p = predict_proba(weights, X, feat)
    pred = p > thr
    return float((pred & (y == 1)).sum() / (y == 1).sum())
