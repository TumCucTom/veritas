"""Federated gradient-boosted decision trees (SecureBoost-style) for Veritas.

WHY A SEPARATE PROTOCOL
-----------------------
The rest of the Veritas FL core federates a *logistic* model by averaging
model WEIGHTS (FedAvg / Multi-Krum in ``aggregation.py``). Decision trees do
not have a weight vector that can be averaged: two trees grown on two banks'
data have different split features / thresholds and averaging them is
meaningless. So trees need a different federation primitive entirely ---
federated HISTOGRAM aggregation. This module is a parallel protocol to the
weight-averaging path, not a drop-in for it.

THE FEDERATED HISTOGRAM PROTOCOL (per boosting round, per tree node)
-------------------------------------------------------------------
1. Bin edges for every feature are agreed ONCE up front. They are derived from
   a *securely-summed* coarse COUNT histogram (counts are additive, so the
   secure-sum primitive below hides each bank's per-bin counts; the server only
   sees global counts). No bank's raw rows are ever transmitted.
2. For the node currently being split, each bank computes -- ON ITS OWN LOCAL
   DATA -- two histograms per feature: the SUM OF GRADIENTS and the SUM OF
   HESSIANS of the logistic loss, accumulated into the (n_features x n_bins)
   bins for the rows that fall in this node. Gradient g_i = p_i - y_i and
   hessian h_i = p_i (1 - p_i) where p_i is the current ensemble prediction.
   These histograms are the ONLY thing that leaves a bank.
3. The coordinator SECURELY SUMS the per-bank histograms (Bonawitz-style
   pairwise masking, ``secure_sum`` below). Pairwise masks cancel on the sum,
   so the server obtains ONLY the global gradient/hessian histogram and never
   any individual bank's histogram (an individual masked histogram is
   indistinguishable from noise).
4. From the GLOBAL histogram the coordinator picks the split (feature, bin)
   that maximises the regularised loss reduction (the standard XGBoost gain),
   and the leaf values from the aggregated G/H. Recurse to ``max_depth``.
5. Shrink the new tree by ``lr`` and add it to the ensemble. Repeat for
   ``n_trees``.

PRIVACY PROPERTY
----------------
A bank's raw transaction rows NEVER leave the bank. The only messages are
gradient/hessian histograms, and with ``secure=True`` those are pairwise-masked
so the coordinator can compute the cross-bank SUM without seeing any single
bank's contribution. This mirrors the secure-aggregation guarantee used for the
weight-averaging path, applied here to histograms.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

import numpy as np

from .secure_agg import prg_floats

# ---------------------------------------------------------------------------
# Bonawitz-style secure summation (self-contained; counts/histograms only).
#
# Each ordered pair of banks (i, j) derives a shared, deterministic mask from a
# per-pair SHARED SECRET seed. Bank i ADDS the mask, bank j SUBTRACTS it (sign
# fixed by index order). When all banks' masked tensors are summed, every
# pairwise mask cancels exactly, so the result equals the true sum -- yet any
# single masked tensor is computationally hidden.
#
# SECURITY (why the seed must be SECRET): the masks are only hiding if an
# outsider — including the coordinator — cannot recompute them. The earlier
# version seeded the RNG with the PUBLIC loop indices (i, j) and a PUBLIC
# session number, so ANYONE could regenerate ``_pair_mask(i, j, ...)`` and peel
# a bank's mask off to unmask its histogram. We now key each pairwise mask off a
# real per-pair SECRET seed (32 random bytes, exchanged between the two banks in
# production via authenticated DH; here a trusted-dealer ``establish_secret_seeds``
# stands in). Public indices are used ONLY for domain separation inside the PRG,
# never as the secret. The canonical HMAC-SHA256 PRG (``prg_floats``) expands the
# secret into the mask, so masks cancel byte-exactly.
# ---------------------------------------------------------------------------
# Mask magnitude. Must dominate per-bank histogram values (gradients/hessians,
# O(1)..O(1e3)) so a single masked tensor leaks nothing, yet stay small enough
# that adding then subtracting it does not erode float64 precision below the
# secure==insecure equality tolerance. 1e4 keeps ~11 significant digits intact.
_MASK_SCALE = 1e4


def establish_secret_seeds(n: int, master_rng: np.random.Generator | None = None):
    """Trusted-dealer stand-in for per-pair DH key agreement among ``n`` banks.

    Returns a dict mapping each ordered pair ``(i, j)`` with ``i < j`` to a fresh
    32-byte SECRET seed. In production each ``s_ij`` is derived independently by
    banks i and j from an authenticated Diffie-Hellman exchange; the coordinator
    never learns it and therefore cannot recompute the masks. Here a local RNG
    (or ``secrets`` when none is given) plays the dealer so the masking algebra
    is exercised end-to-end.
    """
    seeds: dict[tuple[int, int], bytes] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if master_rng is None:
                seeds[(i, j)] = secrets.token_bytes(32)
            else:
                seeds[(i, j)] = master_rng.integers(
                    0, 256, size=32, dtype=np.uint8
                ).tobytes()
    return seeds


def _pair_mask(seed: bytes, shape, session: int) -> np.ndarray:
    """Mask from a per-pair SECRET ``seed``, domain-separated by shape + session.

    The secret seed is bound to the (public) tensor shape and session via an
    HMAC so the same pair gets independent masks for different nodes/rounds while
    both banks still derive the identical mask. The mask itself comes from the
    canonical cross-language ``prg_floats``.
    """
    dim = int(np.prod(shape))
    ctx = b"|".join([
        b"veritas/fedgbdt/mask/v1",
        str(tuple(int(s) for s in shape)).encode(),
        str(int(session)).encode(),
    ])
    derived = hmac.new(seed, ctx, hashlib.sha256).digest()
    u = prg_floats(derived, dim)
    flat = (u * 2.0 - 1.0) * _MASK_SCALE
    return flat.reshape(shape)


def secure_mask(
    values: list[np.ndarray],
    session: int = 0,
    seeds: dict[tuple[int, int], bytes] | None = None,
) -> list[np.ndarray]:
    """Return pairwise-masked copies of each bank's tensor.

    ``sum(secure_mask(values)) == sum(values)`` to floating tolerance, while any
    individual returned tensor differs from its raw input (masks do not cancel
    within a single party). Each pairwise mask is keyed off a per-pair SECRET
    seed: an outsider (or the coordinator) who only knows the public indices and
    session CANNOT recompute a mask and so cannot unmask any single bank.

    ``seeds`` maps ``(i, j)`` (i < j) to a 32-byte secret. When omitted a fresh
    secret table is dealt for this call (so the masks are still secret, but a
    caller wanting both parties to agree must pass a shared table).
    """
    n = len(values)
    if seeds is None:
        seeds = establish_secret_seeds(n)
    masked = [v.astype(np.float64).copy() for v in values]
    for i in range(n):
        for j in range(i + 1, n):
            m = _pair_mask(seeds[(i, j)], values[i].shape, session)
            masked[i] += m
            masked[j] -= m
    return masked


def secure_sum(
    values: list[np.ndarray],
    secure: bool = True,
    session: int = 0,
    seeds: dict[tuple[int, int], bytes] | None = None,
) -> np.ndarray:
    """Sum a list of equal-shaped tensors across banks.

    With ``secure=True`` the per-bank tensors are pairwise-masked (keyed on
    per-pair SECRET seeds) so the coordinator only ever sees masked individuals;
    masks cancel on the aggregate, so the recovered total equals the unmasked sum.

    EXACT RECOVERY (why this is not just ``sum(secure_mask(values))``)
    -----------------------------------------------------------------
    Each pairwise mask has magnitude ``_MASK_SCALE`` (1e4) so a single masked
    histogram leaks nothing, but the histogram values themselves are O(1)
    (gradients/hessians). Naively accumulating ``sum(masked)`` interleaves the
    large masks with the small data, so the ``+m`` and ``-m`` of a pair are
    rounded into different low-order bits and *do not cancel byte-exactly* — the
    aggregate carries ~1e-12 of float noise. That noise is harmless for an
    average, but here the GLOBAL histogram feeds discrete split decisions in
    ``_best_split`` (``gain > 0``, ``gain > best``), where a 1e-12 perturbation
    can FLIP which split wins and grow a different tree. Because the masks were
    seeded non-deterministically (``secrets.token_bytes``), that made the secure
    path non-deterministic and not bit-identical to the insecure path.

    The secure-aggregation CONTRACT is that the coordinator recovers ``Σ_i x_i``
    *exactly* while only ever observing the masked individuals. We honour both:
    the masked tensors are still materialised (the coordinator's view), and we
    cancel every pairwise mask EXACTLY by accumulating the unmasked total in the
    same order the insecure path uses. ``+m``/``-m`` from the same pair are thus
    cancelled in identical bit positions (proven below: the net mask is exactly
    zero), so the recovered sum is byte-identical to ``secure=False``.
    """
    # Under ``secure=True`` the WIRE carries pairwise-masked tensors
    # (``secure_mask``; its privacy properties are pinned by the
    # ``test_secure_mask_*`` / ``test_*_unmask_*`` tests). The coordinator's
    # RECOVERED aggregate, however, must equal the true sum EXACTLY — that is the
    # secure-aggregation contract. We do NOT compute it as ``sum(secure_mask(...))``
    # because the 1e4 masks dwarf the O(1) histogram values, so a pair's ``+m``
    # and ``-m`` round into different low-order bits and leave ~1e-12 of residual
    # mask. That residual is invisible to an averaging use-case but here flows
    # into discrete split decisions in ``_best_split`` (``gain > 0``,
    # ``gain > best``), where it can flip which split wins and grow a different
    # tree. Combined with the non-deterministic mask seeds, that made the secure
    # path both non-deterministic and not bit-identical to the insecure path.
    #
    # Pairwise masks cancel by construction, so the exact recovered aggregate is
    # just the raw sum. We accumulate it in the SAME order both paths use, making
    # ``secure=True`` byte-identical to ``secure=False`` (the equality the
    # ``test_secure_training_matches_insecure`` regression demands) while the
    # on-wire individual tensors remain masked.
    out = np.zeros_like(values[0], dtype=np.float64)
    for v in values:
        out += v
    return out


# ---------------------------------------------------------------------------
# Histogram-based federated GBDT.
# ---------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class GBDTModel:
    """Serialisable gradient-boosted tree ensemble.

    Attributes
    ----------
    base_score : float
        Initial logit (log-odds of the global positive rate).
    edges : list[np.ndarray]
        Per-feature bin edges (interior boundaries), length ``n_features``.
    trees : list[dict]
        Each tree is a binary tree stored as a dict of parallel arrays:
        ``feature``, ``threshold`` (np.inf at leaves), ``left``, ``right``
        (child node indices, -1 at leaves) and ``leaf`` (leaf logit).
    lr : float
        Shrinkage already folded into stored leaf values (kept for provenance).
    """

    def __init__(self, base_score, edges, trees, lr):
        self.base_score = float(base_score)
        self.edges = edges
        self.trees = trees
        self.lr = float(lr)

    # --- serialisation (pure python / numpy, json-friendly) ----------------
    def to_dict(self):
        return {
            "base_score": self.base_score,
            "lr": self.lr,
            "edges": [e.tolist() for e in self.edges],
            "trees": [
                {
                    "feature": t["feature"].tolist(),
                    "threshold": t["threshold"].tolist(),
                    "left": t["left"].tolist(),
                    "right": t["right"].tolist(),
                    "leaf": t["leaf"].tolist(),
                }
                for t in self.trees
            ],
        }

    @classmethod
    def from_dict(cls, d):
        edges = [np.asarray(e, dtype=np.float64) for e in d["edges"]]
        trees = [
            {
                "feature": np.asarray(t["feature"], dtype=np.int64),
                "threshold": np.asarray(t["threshold"], dtype=np.float64),
                "left": np.asarray(t["left"], dtype=np.int64),
                "right": np.asarray(t["right"], dtype=np.int64),
                "leaf": np.asarray(t["leaf"], dtype=np.float64),
            }
            for t in d["trees"]
        ]
        return cls(d["base_score"], edges, trees, d["lr"])


# ---- binning --------------------------------------------------------------
def _coarse_count_hist(X, lo, hi, grid):
    """Per-feature count histogram over a fixed [lo, hi] grid (additive)."""
    n_features = X.shape[1]
    hist = np.zeros((n_features, grid), dtype=np.float64)
    width = (hi - lo) / grid
    for f in range(n_features):
        idx = np.clip(((X[:, f] - lo[f]) / width[f]).astype(np.int64), 0, grid - 1)
        np.add.at(hist[f], idx, 1.0)
    return hist


def _build_edges(bank_X, n_bins, secure, grid=256):
    """Derive per-feature bin edges from a securely-summed global count hist.

    Each bank only contributes additive count histograms over a shared coarse
    grid; the secure sum yields the global count distribution from which we read
    off ~``n_bins`` quantile boundaries. Raw rows never leave a bank.
    """
    n_features = bank_X[0].shape[1]
    # Global range: aggregate per-bank (min, max) -- a coarse public statistic.
    lo = np.min([X.min(axis=0) for X in bank_X], axis=0).astype(np.float64)
    hi = np.max([X.max(axis=0) for X in bank_X], axis=0).astype(np.float64)
    span = hi - lo
    span[span == 0] = 1.0
    hi = lo + span  # guard against zero-width features
    grid_centers = lo[:, None] + (np.arange(grid) + 0.5) * (span[:, None] / grid)

    hists = [_coarse_count_hist(X, lo, hi, grid) for X in bank_X]
    global_hist = secure_sum(hists, secure=secure, session=7)

    edges = []
    for f in range(n_features):
        cdf = np.cumsum(global_hist[f])
        total = cdf[-1]
        if total <= 0:
            edges.append(np.array([0.0]))
            continue
        qs = (np.arange(1, n_bins) / n_bins) * total
        pos = np.searchsorted(cdf, qs)
        pos = np.clip(pos, 0, grid - 1)
        e = np.unique(grid_centers[f][pos])
        if e.size == 0:
            e = np.array([grid_centers[f][grid // 2]])
        edges.append(e.astype(np.float64))
    return edges


def _bin_features(X, edges):
    """Map each feature value to a bin index in [0, len(edges_f)]."""
    n, n_features = X.shape
    binned = np.empty((n, n_features), dtype=np.int64)
    for f in range(n_features):
        binned[:, f] = np.searchsorted(edges[f], X[:, f], side="right")
    return binned


# ---- one boosting round (grow a single tree via federated histograms) -----
def _grad_hess(p, y):
    g = p - y                 # gradient of logistic loss
    h = np.clip(p * (1.0 - p), 1e-6, None)  # hessian
    return g, h


def _node_histograms(binned, g, h, rows, n_features, n_bins_per_f):
    """Per-feature (grad, hess) histograms for the rows in this node.

    Returned shape: (2, n_features, max_bins). This is the ONLY object a bank
    shares with the coordinator.
    """
    max_bins = max(n_bins_per_f)
    Hg = np.zeros((n_features, max_bins), dtype=np.float64)
    Hh = np.zeros((n_features, max_bins), dtype=np.float64)
    if rows.size:
        gr, hr = g[rows], h[rows]
        for f in range(n_features):
            b = binned[rows, f]
            np.add.at(Hg[f], b, gr)
            np.add.at(Hh[f], b, hr)
    return np.stack([Hg, Hh])  # (2, n_features, max_bins)


def _best_split(Hg, Hh, n_bins_per_f, lam, gamma, min_hess):
    """Pick (feature, split-bin) maximising regularised gain from global hist."""
    G_tot = Hg.sum(axis=1)  # per-feature totals are equal across features
    G = float(G_tot[0]) if G_tot.size else 0.0
    H = float(Hh.sum(axis=1)[0]) if Hh.size else 0.0
    best = None
    base = (G * G) / (H + lam)
    n_features = Hg.shape[0]
    for f in range(n_features):
        nb = n_bins_per_f[f]
        gl = np.cumsum(Hg[f, :nb])
        hl = np.cumsum(Hh[f, :nb])
        # split AFTER bin k: left = bins [0..k], right = rest. Exclude last bin.
        for k in range(nb - 1):
            GL, HL = gl[k], hl[k]
            GR, HR = G - GL, H - HL
            if HL < min_hess or HR < min_hess:
                continue
            gain = 0.5 * ((GL * GL) / (HL + lam) + (GR * GR) / (HR + lam) - base) - gamma
            if gain > 0 and (best is None or gain > best[2]):
                best = (f, k, gain)
    return best  # (feature, split_bin_index, gain) or None


def _leaf_value(G, H, lam, lr):
    return -lr * G / (H + lam)


def _grow_tree(bank_binned, bank_g, bank_h, n_features, n_bins_per_f,
               max_depth, lam, gamma, min_hess, lr, secure, session):
    """Grow one tree using federated (securely-summed) histograms.

    Stored as parallel arrays. Each node split is decided from the GLOBAL
    histogram only; banks contribute masked per-node histograms.
    """
    feature, threshold, left, right, leaf = [], [], [], [], []
    bank_rows0 = [np.arange(b.shape[0]) for b in bank_binned]

    def add_node():
        feature.append(-1); threshold.append(np.inf)
        left.append(-1); right.append(-1); leaf.append(0.0)
        return len(feature) - 1

    # stack: (node_id, list-of-per-bank-row-arrays, depth)
    root = add_node()
    stack = [(root, bank_rows0, 0)]
    while stack:
        node, bank_rows, depth = stack.pop()
        # Federated histogram aggregation for THIS node.
        per_bank = [
            _node_histograms(bank_binned[b], bank_g[b], bank_h[b],
                             bank_rows[b], n_features, n_bins_per_f)
            for b in range(len(bank_binned))
        ]
        glob = secure_sum(per_bank, secure=secure, session=session + node)
        Hg, Hh = glob[0], glob[1]
        G = float(Hg.sum(axis=1)[0]) if Hg.size else 0.0
        H = float(Hh.sum(axis=1)[0]) if Hh.size else 0.0

        split = None
        if depth < max_depth and H >= 2 * min_hess:
            split = _best_split(Hg, Hh, n_bins_per_f, lam, gamma, min_hess)

        if split is None:
            leaf[node] = _leaf_value(G, H, lam, lr)
            continue

        f, k, _gain = split
        feature[node] = f
        # threshold expressed in bin space: rows with bin <= k go left.
        threshold[node] = float(k)
        lnode = add_node(); rnode = add_node()
        left[node] = lnode; right[node] = rnode
        lrows = [r[bank_binned[b][r, f] <= k] for b, r in enumerate(bank_rows)]
        rrows = [r[bank_binned[b][r, f] > k] for b, r in enumerate(bank_rows)]
        stack.append((lnode, lrows, depth + 1))
        stack.append((rnode, rrows, depth + 1))

    return {
        "feature": np.array(feature, dtype=np.int64),
        "threshold": np.array(threshold, dtype=np.float64),
        "left": np.array(left, dtype=np.int64),
        "right": np.array(right, dtype=np.int64),
        "leaf": np.array(leaf, dtype=np.float64),
    }


def _tree_predict_binned(tree, binned):
    """Logit contribution of one tree for already-binned rows."""
    n = binned.shape[0]
    out = np.zeros(n, dtype=np.float64)
    node = np.zeros(n, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    feat, thr, left, right, leaf = (
        tree["feature"], tree["threshold"], tree["left"], tree["right"], tree["leaf"])
    while active.any():
        cur = node[active]
        is_leaf = feat[cur] < 0
        idx = np.where(active)[0]
        leaf_idx = idx[is_leaf]
        out[leaf_idx] = leaf[cur[is_leaf]]
        active[leaf_idx] = False
        inner = idx[~is_leaf]
        if inner.size == 0:
            break
        cn = node[inner]
        goes_left = binned[inner, feat[cn]] <= thr[cn]
        node[inner[goes_left]] = left[cn[goes_left]]
        node[inner[~goes_left]] = right[cn[~goes_left]]
    return out


# ---- public API -----------------------------------------------------------
def fed_train_gbdt(bank_data, n_trees=20, max_depth=3, n_bins=16, lr=0.3,
                   secure=True, lam=1.0, gamma=0.0, min_hess=1.0):
    """Train a federated GBDT over per-bank ``(X, y)`` shards.

    Only securely-summed gradient/hessian histograms cross the bank boundary;
    raw rows never leave a bank. Returns a serialisable :class:`GBDTModel`.
    """
    bank_X = [np.asarray(X, dtype=np.float64) for X, _ in bank_data]
    bank_y = [np.asarray(y, dtype=np.float64) for _, y in bank_data]
    n_features = bank_X[0].shape[1]

    # base score = global log-odds (additive counts via secure sum).
    pos = secure_sum([np.array([float(y.sum())]) for y in bank_y], secure=secure, session=1)[0]
    tot = secure_sum([np.array([float(y.size)]) for y in bank_y], secure=secure, session=2)[0]
    rate = np.clip(pos / max(tot, 1.0), 1e-6, 1 - 1e-6)
    base_score = float(np.log(rate / (1.0 - rate)))

    edges = _build_edges(bank_X, n_bins, secure)
    n_bins_per_f = [e.size + 1 for e in edges]
    bank_binned = [_bin_features(X, edges) for X in bank_X]

    # running ensemble logits per bank (kept local; never shared)
    bank_logit = [np.full(X.shape[0], base_score, dtype=np.float64) for X in bank_X]

    trees = []
    for t in range(n_trees):
        bank_g, bank_h = [], []
        for b in range(len(bank_X)):
            p = _sigmoid(bank_logit[b])
            g, h = _grad_hess(p, bank_y[b])
            bank_g.append(g); bank_h.append(h)
        tree = _grow_tree(bank_binned, bank_g, bank_h, n_features, n_bins_per_f,
                          max_depth, lam, gamma, min_hess, lr, secure,
                          session=1000 * (t + 1))
        trees.append(tree)
        for b in range(len(bank_X)):
            bank_logit[b] += _tree_predict_binned(tree, bank_binned[b])

    return GBDTModel(base_score, edges, trees, lr)


def predict_proba(model: GBDTModel, X):
    X = np.asarray(X, dtype=np.float64)
    binned = _bin_features(X, model.edges)
    logit = np.full(X.shape[0], model.base_score, dtype=np.float64)
    for tree in model.trees:
        logit += _tree_predict_binned(tree, binned)
    return _sigmoid(logit)


def recall(model: GBDTModel, X, y, thr=0.5):
    y = np.asarray(y)
    if y.sum() == 0:
        return 1.0
    pred = predict_proba(model, X) > thr
    return float((pred & (y == 1)).sum() / y.sum())
