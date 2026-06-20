"""LIVE-LOOP ENSEMBLE ADAPTER — drop the Veritas stacked ensemble into the
node<->plane federated loop *as if it were a single logistic model*.

The problem this solves
-----------------------
The live federation contract today is the one in ``veritas_core.model`` (a flat
logistic regression):

    init_weights(FEATURE_DIM) -> w           # one flat vector
    train_local(w, X, y, ...) -> w           # local round
    predict_proba(w, X)       -> probs        # inference
    recall(w, X, y)           -> float

The node calls exactly these, the plane robust-aggregates the ONE flat vector
(Multi-Krum / DP), and ``/predict`` runs it. ``LiveEnsemble`` exposes the
*ensemble* behind the SAME single-flat-vector interface so swapping it in is
mechanical and the federation contract is UNCHANGED: its parameters are the
CONCATENATION of each base model's flat weight vector plus the meta-learner's,
packed into one fixed-dim vector. A *list* of these packed vectors drops
straight into ``aggregation.fedavg`` / ``aggregation.multi_krum`` and a single
delta into ``dp.privatize`` with no changes — they only ever see a fixed-dim
1-D float array.

Base set for the live loop (fast enough for an interactive demo)
----------------------------------------------------------------
    * logistic   (model.py)      -> linear slice          [view: X tabular]
    * MLP        (mlp.py)        -> XOR / interaction      [view: X tabular]
    * embeddings (embeddings.py) -> categorical corridor   [view: X_cat + X]

combined by a federated LOGISTIC meta-learner over the three base probabilities.
No GRU / GBDT here (those need their own protocols and are slow); this trio
already covers the three heterogeneous fraud typologies in ``make_live_data`` so
the ensemble demonstrably beats logistic alone.

The ``data`` container
----------------------
A plain dict (see :class:`LiveData` for a typed convenience wrapper) with:

    "X"     : (n, FEATURE_DIM) float64   dense tabular view  (logistic / MLP / emb-dense)
    "X_cat" : (n, 3)           int64     categorical view    [merchant, country, channel]

Labels ``y`` travel SEPARATELY (as in ``model.py``) — ``predict_proba`` /
``predict_one`` never need them; ``train_local`` / ``recall`` take ``y`` as an
argument. The same container is consumed by every method.

Packed weight-vector layout (fixed dim, identical across banks)
---------------------------------------------------------------
    [ logistic_block | mlp_block | emb_block | meta_block ]

      logistic_block : model.init_weights(FEATURE_DIM)         -> FEATURE_DIM+1
      mlp_block      : mlp.init_weights(FEATURE_DIM, HIDDEN)   -> mlp.weight_dim(...)
      emb_block      : emb.init_weights(CARD, emb_dim, ...)    -> emb.weight_dim(...)
      meta_block     : logistic.init_weights(3)               -> 4  (3 base probs + bias)

``weight_dim()`` is the total length; ``block_slices()`` gives each block's
[start, end) index range. The blocks are contiguous, non-overlapping and cover
exactly ``weight_dim()``.

WHY per-block DP clipping matters
---------------------------------
The blocks are HETEROGENEOUS in scale and dimensionality: the logistic block is
~11 numbers, the MLP / embedding blocks are hundreds, and a typical update's L2
norm is dominated by the largest block. A single GLOBAL clip on the whole packed
delta would therefore be set by (and distort) the big blocks — effectively
zeroing the contribution of the small logistic / meta blocks, or conversely
leaving the big blocks barely clipped. ``block_slices()`` lets the node clip
EACH block's delta to its own norm budget (and add per-block noise) so every
sub-model gets a faithful, comparably-scaled private update. See
:func:`clip_blocks` for a reference per-block clip.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import model as logistic
from . import mlp
from . import embeddings as emb
from .data import FEATURE_DIM
from .dp import add_noise, clip_update


# ---------------------------------------------------------------------------
# Fixed config — identical across every bank so the packed vector lines up.
# ---------------------------------------------------------------------------
MLP_HIDDEN = (12, 6)          # small hidden dims: fast per-round on every node
EMB_DIM = 4
EMB_HIDDEN = 12
# Shared categorical schema (must match make_live_data): [merchant, country, channel]
N_MERCHANT, N_COUNTRY, N_CHANNEL = 12, 8, 4
CARD = [N_MERCHANT, N_COUNTRY, N_CHANNEL]
N_BASE = 3                    # logistic + mlp + embeddings


# ---------------------------------------------------------------------------
# data container
# ---------------------------------------------------------------------------
@dataclass
class LiveData:
    """Multi-view container the ensemble methods consume.

    ``X``     : (n, FEATURE_DIM) float64 tabular features.
    ``X_cat`` : (n, 3)           int64   categorical indices [merchant, country, channel].
    """

    X: np.ndarray
    X_cat: np.ndarray

    def __len__(self):
        return len(self.X)


def _as_views(data):
    """Accept a LiveData, a dict, or a bare tabular array; return (X, X_cat)."""
    if isinstance(data, LiveData):
        X, X_cat = data.X, data.X_cat
    elif isinstance(data, dict):
        X = data["X"]
        X_cat = data.get("X_cat")
    else:  # bare array (robustness for /predict)
        X = np.asarray(data, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_cat = None
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X_cat is None:
        # neutral / default categorical row(s): category 0 everywhere (NOT the
        # mule corridor, which is merchant==3 & country==5).
        X_cat = np.zeros((X.shape[0], len(CARD)), dtype=np.int64)
    X_cat = np.asarray(X_cat, dtype=np.int64)
    if X_cat.ndim == 1:
        X_cat = X_cat.reshape(1, -1)
    return X, X_cat


# ---------------------------------------------------------------------------
# Packed weight-vector layout
# ---------------------------------------------------------------------------
def _block_sizes():
    """Length of each block in the packed vector, in pack order."""
    return {
        "logistic": FEATURE_DIM + 1,
        "mlp": mlp.weight_dim(in_dim=FEATURE_DIM, hidden=MLP_HIDDEN),
        "embeddings": emb.weight_dim(CARD, emb_dim=EMB_DIM,
                                     dense_dim=FEATURE_DIM, hidden=EMB_HIDDEN),
        "meta": N_BASE + 1,
    }


def block_slices():
    """dict name -> (start, end) index range of each model's block.

    Contiguous, non-overlapping, and together covering exactly ``weight_dim()``.
    The node uses these to DP-CLIP PER BLOCK (see module docstring): one global
    clip would distort the heterogeneous blocks.
    """
    sizes = _block_sizes()
    out = {}
    i = 0
    for name in ("logistic", "mlp", "embeddings", "meta"):
        out[name] = (i, i + sizes[name])
        i += sizes[name]
    return out


def weight_dim():
    """Total length of the packed flat vector (fixed; identical across banks)."""
    return sum(_block_sizes().values())


def _split(packed):
    """Slice a packed vector into its four blocks."""
    sl = block_slices()
    return {name: packed[a:b] for name, (a, b) in sl.items()}


def _join(blocks):
    """Concatenate blocks back into one packed vector (pack order)."""
    return np.concatenate([blocks["logistic"], blocks["mlp"],
                           blocks["embeddings"], blocks["meta"]])


# ---------------------------------------------------------------------------
# Single-model interface (mirrors model.py exactly)
# ---------------------------------------------------------------------------
def init_weights(seed=0):
    """Packed flat vector of all base models + meta-learner (fixed dim).

    Identical dimension across banks for the fixed config above. Mirrors
    ``model.init_weights`` but takes only a seed (the dim is implicit).
    """
    blocks = {
        "logistic": logistic.init_weights(FEATURE_DIM),
        "mlp": mlp.init_weights(in_dim=FEATURE_DIM, hidden=MLP_HIDDEN, seed=seed),
        "embeddings": emb.init_weights(CARD, emb_dim=EMB_DIM,
                                       dense_dim=FEATURE_DIM, hidden=EMB_HIDDEN,
                                       seed=seed),
        "meta": logistic.init_weights(N_BASE),
    }
    return _join(blocks)


def _base_probas(blocks, X, X_cat):
    """(n, N_BASE) matrix of each base model's fraud probability."""
    p_log = logistic.predict_proba(blocks["logistic"], X)
    p_mlp = mlp.predict_proba(blocks["mlp"], X, in_dim=FEATURE_DIM,
                              hidden=MLP_HIDDEN)
    p_emb = emb.predict_proba(blocks["embeddings"], X_cat, X, CARD,
                              emb_dim=EMB_DIM, dense_dim=FEATURE_DIM,
                              hidden=EMB_HIDDEN)
    return np.column_stack([p_log, p_mlp, p_emb])


def predict_proba(packed, data):
    """Per-sample fraud probability in [0, 1].

    Unpacks the vector, runs each base model on its view, then the logistic
    meta-learner over the three base probabilities.
    """
    X, X_cat = _as_views(data)
    blocks = _split(packed)
    base_p = _base_probas(blocks, X, X_cat)
    return np.clip(logistic.predict_proba(blocks["meta"], base_p), 0.0, 1.0)


def train_local(packed, data, y, epochs=12, lr=0.2, meta_epochs=120,
                meta_lr=0.3, meta_frac=0.4, seed=0):
    """One local federated round: train each base on its view + the meta on the
    bases' held-out predictions, then repack.

    Uses a per-sample TRAIN / META split (the rows the meta-learner sees are
    rows NO base trained on) to avoid stacking leakage — same device as the
    offline ``ensemble.StackedEnsemble``, applied locally. Returns the updated
    packed vector (same dimension).
    """
    X, X_cat = _as_views(data)
    y = np.asarray(y).astype(np.int64)
    blocks = {k: v.copy() for k, v in _split(packed).items()}
    n = len(y)

    # per-sample disjoint base / meta split (anti-leakage)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_meta = max(1, int(round(n * meta_frac))) if n > 1 else 0
    meta_idx = np.sort(perm[:n_meta])
    base_idx = np.sort(perm[n_meta:]) if n_meta < n else perm
    if base_idx.size == 0:            # degenerate tiny n: train bases on all rows
        base_idx = perm
        meta_idx = perm

    Xb, Xcb, yb = X[base_idx], X_cat[base_idx], y[base_idx]

    # 1. train each base on its VIEW (base rows only)
    blocks["logistic"] = logistic.train_local(blocks["logistic"], Xb, yb,
                                              epochs=epochs, lr=lr)
    blocks["mlp"] = mlp.train_local(blocks["mlp"], Xb, yb, epochs=epochs, lr=lr,
                                    in_dim=FEATURE_DIM, hidden=MLP_HIDDEN)
    blocks["embeddings"] = emb.train_local(blocks["embeddings"], Xcb, Xb, yb,
                                           epochs=epochs, lr=lr,
                                           cardinalities=CARD, emb_dim=EMB_DIM,
                                           dense_dim=FEATURE_DIM,
                                           hidden=EMB_HIDDEN)

    # 2. meta-features = base preds on the held-out META rows (unseen by bases)
    Xm, Xcm, ym = X[meta_idx], X_cat[meta_idx], y[meta_idx]
    meta_X = _base_probas(blocks, Xm, Xcm)
    blocks["meta"] = logistic.train_local(blocks["meta"], meta_X, ym,
                                          epochs=meta_epochs, lr=meta_lr)
    return _join(blocks)


def recall(packed, data, y, thr=0.5):
    """Fraud-detection recall of the ensemble at threshold ``thr``."""
    y = np.asarray(y).astype(np.int64)
    if y.sum() == 0:
        return 1.0
    pred = predict_proba(packed, data) > thr
    return float((pred & (y == 1)).sum() / (y == 1).sum())


def predict_one(packed, features):
    """Score ONE transaction for ``/predict``.

    ``features`` may be:
      * a dict of named tabular features (keys per ``data.py`` FEATURE_DIM order,
        missing keys -> 0.0), optionally also carrying "X_cat" /
        "merchant"/"country"/"channel" for the categorical view;
      * a LiveData / dict with "X" (+ optional "X_cat");
      * a bare FEATURE_DIM array / list (categorical defaults to neutral 0s).

    Builds a one-row multi-view (tabular from the features; neutral categorical
    unless supplied) and returns a single fraud probability in [0, 1].
    """
    if isinstance(features, dict) and "X" not in features:
        # named-feature dict -> build a FEATURE_DIM row in data.py order
        names = ["amount", "oldOrig", "newOrig", "oldDest", "newDest",
                 "accountAge", "velocity", "fanout", "isTransfer", "campaignSig"]
        row = np.array([float(features.get(nm, 0.0)) for nm in names],
                       dtype=np.float64).reshape(1, -1)
        cat = features.get("X_cat")
        if cat is None and any(k in features for k in ("merchant", "country",
                                                       "channel")):
            cat = [[int(features.get("merchant", 0)),
                    int(features.get("country", 0)),
                    int(features.get("channel", 0))]]
        data = {"X": row, "X_cat": cat}
    else:
        data = features
    return float(predict_proba(packed, data)[0])


# ---------------------------------------------------------------------------
# Per-block DP clip (reference helper for the node)
# ---------------------------------------------------------------------------
def clip_blocks(delta, max_norms):
    """Clip EACH block of a packed delta to its own L2 norm, then repack.

    ``max_norms`` : float (same budget for every block) or dict name->budget.
    Returns a vector of the SAME dim as ``delta``. This is the per-block analogue
    of ``dp.clip_update``; doing it per block (rather than one global clip) keeps
    the heterogeneous blocks comparably scaled — see module docstring.
    """
    sl = block_slices()
    out = np.array(delta, dtype=np.float64, copy=True)
    for name, (a, b) in sl.items():
        budget = max_norms[name] if isinstance(max_norms, dict) else max_norms
        seg = out[a:b]
        nrm = float(np.linalg.norm(seg))
        if nrm > budget and nrm > 0.0:
            out[a:b] = seg * (budget / nrm)
    return out


# ---------------------------------------------------------------------------
# Canonical wire dimension + per-block DP budgets (node federation contract)
# ---------------------------------------------------------------------------
# Convenience alias of ``weight_dim()`` so the node and the dimension-agnostic
# plane can derive the expected wire length without a magic number.
LIVE_ENSEMBLE_DIM: int = weight_dim()

# Per-block L2 clipping budgets for the packed delta. The blocks are
# HETEROGENEOUS in scale/dimensionality (see module docstring): the MLP and
# embedding blocks carry hundreds of params (large L2) while the logistic and
# meta blocks carry ~11 and 4 (tiny L2). A single GLOBAL clip would be set by —
# and crush — the big blocks, zeroing the small ones. These are the REAL
# clipping radii; the noise scale below is calibrated to THEM so the implied
# (eps, delta) matches what is actually enforced on the wire.
BLOCK_BUDGETS: dict[str, float] = {
    "logistic": 3.0,
    "mlp": 6.0,
    "embeddings": 8.0,
    "meta": 3.0,
}


def block_slice(name: str) -> slice:
    """``slice`` object for one named block of the packed vector."""
    a, b = block_slices()[name]
    return slice(a, b)


def genesis_weights() -> np.ndarray:
    """A zero live-ensemble vector of the canonical dimension.

    Used as the federation starting point; the dimension-agnostic plane derives
    the expected wire length from this genesis/global model (no magic number).
    """
    return np.zeros(LIVE_ENSEMBLE_DIM, dtype=np.float64)


def noise_sensitivity(budgets: dict[str, float] | None = None) -> float:
    """Total L2 clipping sensitivity of one per-block-clipped update.

    Blocks are clipped INDEPENDENTLY, so the worst-case L2 of the whole
    concatenated update is ``sqrt(sum(budget_b^2))``. The Gaussian mechanism's
    per-coordinate std is ``sigma * sensitivity`` — calibrating to THIS (rather
    than a stale single ``max_norm``) makes the implied privacy guarantee honest.
    """
    budgets = budgets or BLOCK_BUDGETS
    return math.sqrt(sum(v * v for v in budgets.values()))


def privatize(update, sigma, *, budgets: dict[str, float] | None = None,
              rng: np.random.Generator | None = None):
    """Per-block-clip ``update`` then add Gaussian noise of the right scale.

    Clipping is per-block (``budgets``, via :func:`clip_blocks`). The noise
    standard deviation is ``sigma * noise_sensitivity(budgets)`` so it is
    calibrated to the ACTUAL total L2 sensitivity the clipping allows, not a
    stale constant. ``rng`` follows SPEC-DP: the default is a CSPRNG-seeded
    generator (via ``veritas_core.dp.add_noise``); pass an explicit seeded
    generator only in tests.
    """
    budgets = budgets or BLOCK_BUDGETS
    update = np.asarray(update, dtype=np.float64)
    clipped = clip_blocks(update, budgets)
    sensitivity = noise_sensitivity(budgets)
    # add_noise uses sigma*max_norm as the per-coordinate std; pass the real
    # total sensitivity as max_norm so the noise matches the enforced clipping.
    return add_noise(clipped, sigma, sensitivity, rng)


# ---------------------------------------------------------------------------
# Multi-view data generator (heterogeneous fraud so ensemble beats logistic)
# ---------------------------------------------------------------------------
def make_live_data(n, fraud_rate=0.05, seed=0, inject_campaign=False):
    """Multi-view fraud sample with HETEROGENEOUS typologies.

    Each account carries the ``X`` (tabular) and ``X_cat`` (categorical) views and
    ONE shared label. Fraud is a UNION of three decorrelated typologies so the
    ensemble (logistic + MLP + embeddings) demonstrably beats logistic alone:

        1. LINEAR slice       -> X[:,0]+X[:,6]+X[:,7] margin   (logistic)
        2. XOR / interaction  -> sign(X[:,3]) != sign(X[:,4])  (MLP)
        3. CATEGORICAL corridor -> merchant==3 AND country==5  (embeddings)

    The views are decorrelated: a corridor fraud looks benign on the tabular
    view, an XOR fraud is invisible to a linear margin, etc. So logistic recovers
    only typology 1, while the ensemble unions all three.

    Parameters
    ----------
    n : int                target number of rows.
    fraud_rate : float     per-typology base rate (so ~3x overall fraud).
    seed : int
    inject_campaign : bool if True, append a "safe-account mule" campaign block
        (mirrors ``data.inject_campaign``): rows that look LEGIT on the generic
        fraud axes but carry the campaign signature in feature [-1]. Lets the
        node's siloed-vs-federated story persist.

    Returns
    -------
    (data, y) : (LiveData, (m,) int64)   where m == n (+ campaign rows).
    """
    rng = np.random.default_rng(seed)

    X = rng.normal(0.0, 1.0, (n, FEATURE_DIM))
    X[:, -1] = 0.0  # campaign-signature feature off by default
    merchant = rng.integers(0, N_MERCHANT, n)
    country = rng.integers(0, N_COUNTRY, n)
    channel = rng.integers(0, N_CHANNEL, n)
    y = np.zeros(n, dtype=np.int64)

    # ---- typology 1: LINEAR slice (logistic) --------------------------
    lin = rng.random(n) < fraud_rate
    X[lin, 0] += 1.8
    X[lin, 6] += 1.5
    X[lin, 7] += 1.5
    y[lin] = 1

    # ---- typology 2: XOR / interaction (MLP) --------------------------
    xor = rng.random(n) < fraud_rate
    a = rng.choice([-1.0, 1.0], int(xor.sum()))
    X[xor, 3] = a * 1.6
    X[xor, 4] = -a * 1.6          # opposite sign -> XOR-positive
    y[xor] = 1
    # XOR-negative decoys in benign rows (same sign) so the margin is non-linear
    benign = np.where(y == 0)[0]
    dec = benign[rng.random(benign.size) < 0.15]
    s = rng.choice([-1.0, 1.0], dec.size)
    X[dec, 3] = s * 1.6
    X[dec, 4] = s * 1.6          # same sign -> stays benign

    # ---- typology 3: CATEGORICAL corridor (embeddings) ----------------
    cor = rng.random(n) < fraud_rate
    merchant[cor] = 3
    country[cor] = 5
    y[cor] = 1
    # plant corridor values individually in benign rows so neither value alone
    # is fraudulent — only the co-occurrence is.
    benign2 = np.where(y == 0)[0]
    half = benign2[rng.random(benign2.size) < 0.2]
    merchant[half[: half.size // 2]] = 3
    country[half[half.size // 2:]] = 5

    X_cat = np.stack([merchant, country, channel], axis=1).astype(np.int64)
    X = X.astype(np.float64)

    if inject_campaign:
        n_camp = max(1, int(round(n * fraud_rate)))
        crng = np.random.default_rng(seed + 999)
        Xc = crng.normal(0.0, 1.0, (n_camp, FEATURE_DIM))
        Xc[:, 0] = -0.5; Xc[:, 6] = -0.5; Xc[:, 7] = -0.5
        Xc[:, 5] = -2.0; Xc[:, -1] = 1.5      # campaign signature
        cat_c = np.stack([
            crng.integers(0, N_MERCHANT, n_camp),
            crng.integers(0, N_COUNTRY, n_camp),
            crng.integers(0, N_CHANNEL, n_camp),
        ], axis=1).astype(np.int64)
        X = np.vstack([X, Xc.astype(np.float64)])
        X_cat = np.vstack([X_cat, cat_c])
        y = np.concatenate([y, np.ones(n_camp, dtype=np.int64)])

    return LiveData(X=X, X_cat=X_cat), y


def partition(data, y, n_banks, seed=0):
    """Split a (data, y) sample into ``n_banks`` per-bank (data, y) shards.

    Lets the node run the siloed-vs-federated story across banks. Returns a list
    of (LiveData, y) tuples.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    shards = np.array_split(perm, n_banks)
    out = []
    for idx in shards:
        idx = np.sort(idx)
        out.append((LiveData(X=data.X[idx], X_cat=data.X_cat[idx]), y[idx]))
    return out
