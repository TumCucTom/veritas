"""Byzantine-robust aggregation for the Veritas FL control plane.

This module provides *principled* robust aggregators that replace the
hand-rolled 10x-median heuristic in the control plane. The threat model
(see ``veritas_core.attack``) is gradient/weight poisoning: a fraction of
member banks submit updates that sign-flip and amplify the honest signal in
order to drag the global model toward whitelisting fraud.

All functions are pure numpy and operate on a list of update vectors (each a
flat ``np.ndarray`` of identical shape). They return a single aggregate of the
same shape, and the selecting aggregators additionally return which members
were kept / rejected so the caller can fire ``attack_detected`` and report the
offending members.

Definitions used throughout
---------------------------
* ``n``  : number of members / updates.
* ``f``  : assumed number of Byzantine (malicious) members = ``n_byzantine``.
* Krum score of update ``i`` : sum of the squared Euclidean distances from
  ``i`` to its ``n - f - 2`` nearest other updates. Honest updates cluster, so
  they have small scores; a lone attacker is far from everyone and scores high.

Robustness guarantees (informal)
---------------------------------
* ``trimmed_mean`` with ``trim_ratio`` chosen so that the number trimmed per
  side ``>= f`` is robust: every coordinate has the ``f`` most extreme values
  on each side discarded, so no set of ``f`` colluding attackers can move a
  coordinate beyond the range of the honest values.
* ``coordinate_median`` tolerates up to ``< n/2`` arbitrary corruptions per
  coordinate.
* ``multi_krum_select`` / ``bulyan`` give the classic Krum / Bulyan Byzantine
  guarantees and require ``n >= 2f + 3`` (Krum) and ``n >= 4f + 3`` (Bulyan).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# 1. Norm clipping -- the cheap robust baseline.
# --------------------------------------------------------------------------- #
def norm_clip(updates, max_norm):
    """Clip each update's L2 norm to ``max_norm``.

    Defends against magnitude / scaling attacks (the ``amplify`` half of the
    poisoning model): an attacker that multiplies the honest direction by a
    large scale is rescaled back to ``max_norm`` so it can no longer dominate
    the average. Direction is preserved; only oversized magnitudes shrink.

    Parameters
    ----------
    updates : sequence of np.ndarray
        Member updates, all the same shape.
    max_norm : float
        Maximum allowed L2 norm. Updates with a smaller norm are unchanged.

    Returns
    -------
    list of np.ndarray
        The clipped updates (new arrays; inputs are not mutated).
    """
    if max_norm is None or max_norm <= 0:
        raise ValueError("max_norm must be a positive float")
    clipped = []
    for u in updates:
        u = np.asarray(u, dtype=float)
        norm = float(np.linalg.norm(u))
        if norm > max_norm and norm > 0.0:
            u = u * (max_norm / norm)
        else:
            u = u.copy()
        clipped.append(u)
    return clipped


# --------------------------------------------------------------------------- #
# 2. Coordinate-wise trimmed mean.
# --------------------------------------------------------------------------- #
def trimmed_mean(updates, trim_ratio):
    """Coordinate-wise trimmed mean.

    For each coordinate independently, sort the ``n`` values, drop the highest
    ``k`` and lowest ``k`` (where ``k = floor(trim_ratio * n)``), and average
    the remaining ``n - 2k`` values. This is Byzantine-robust whenever
    ``k >= f`` because the ``f`` colluding attackers can occupy at most the
    ``f`` extreme slots on one side, which are discarded.

    On all-honest, low-variance input this reduces to (approximately) the mean.

    Parameters
    ----------
    updates : sequence of np.ndarray
        Member updates, all the same shape.
    trim_ratio : float
        Fraction in ``[0, 0.5)`` to drop from *each* tail per coordinate.

    Returns
    -------
    np.ndarray
        The coordinate-wise trimmed mean, same shape as a single update.
    """
    if not 0.0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5)")
    U = np.stack([np.asarray(u, dtype=float) for u in updates])
    n = U.shape[0]
    k = int(np.floor(trim_ratio * n))
    if k == 0:
        return U.mean(axis=0)
    if n - 2 * k <= 0:
        # Degenerate: nothing would survive. Fall back to the median, which is
        # the limiting case of full trimming.
        return np.median(U, axis=0)
    # Sort each coordinate ascending, then slice off k from each tail.
    sorted_U = np.sort(U, axis=0)
    kept = sorted_U[k : n - k]
    return kept.mean(axis=0)


# --------------------------------------------------------------------------- #
# 3. Coordinate-wise median.
# --------------------------------------------------------------------------- #
def coordinate_median(updates):
    """Coordinate-wise median -- robust and parameter-free.

    Tolerates up to ``< n/2`` arbitrarily corrupted values per coordinate. On
    all-honest input it equals the median, which is ~mean for symmetric noise.

    Parameters
    ----------
    updates : sequence of np.ndarray

    Returns
    -------
    np.ndarray
        The coordinate-wise median, same shape as a single update.
    """
    U = np.stack([np.asarray(u, dtype=float) for u in updates])
    return np.median(U, axis=0)


# --------------------------------------------------------------------------- #
# Internal: pairwise squared-distance matrix + Krum scores.
# --------------------------------------------------------------------------- #
def _pairwise_sq_dist(U):
    """Symmetric matrix of squared Euclidean distances between rows of ``U``."""
    n = U.shape[0]
    flat = U.reshape(n, -1)
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b  (numerically fine for our scales).
    sq = np.sum(flat * flat, axis=1)
    gram = flat @ flat.T
    dist = sq[:, None] + sq[None, :] - 2.0 * gram
    np.fill_diagonal(dist, 0.0)
    # Clamp tiny negatives from floating point.
    return np.maximum(dist, 0.0)


def _krum_scores(dist, n_byzantine):
    """Krum score per row: sum of squared distances to the ``n-f-2`` nearest
    *other* rows."""
    n = dist.shape[0]
    # Number of neighbours summed in the classic Krum formulation.
    k = max(1, n - n_byzantine - 2)
    scores = np.empty(n, dtype=float)
    for i in range(n):
        # Exclude self (distance 0 at index i) by taking sorted[1 : k+1].
        order = np.sort(dist[i])
        scores[i] = float(order[1 : k + 1].sum())
    return scores


# --------------------------------------------------------------------------- #
# 4. Principled Multi-Krum that SELECTS.
# --------------------------------------------------------------------------- #
def multi_krum_select(updates, n_byzantine, m=None):
    """Principled Multi-Krum selection.

    Computes a Krum score for every update and keeps the ``m`` lowest-scoring
    (most "central") ones, averaging only those. This is the key difference
    from a naive defence that averages everyone after dropping one outlier:
    Multi-Krum *selects* a Byzantine-free majority and discards the rest, which
    is what gives the formal Krum robustness guarantee.

    Requires ``n >= 2f + 3`` for the guarantee to hold; smaller ``n`` still
    runs (scores remain meaningful) but loses the theoretical bound.

    Parameters
    ----------
    updates : sequence of np.ndarray
    n_byzantine : int
        Assumed number of Byzantine members ``f`` (>= 1 supported).
    m : int, optional
        Number of updates to select and average. Defaults to ``n - f``
        (the expected number of honest members), at least 1.

    Returns
    -------
    aggregate : np.ndarray
        Mean of the ``m`` selected updates.
    selected_idx : list of int
        Indices of the selected (kept) updates, in ascending order.
    scores : np.ndarray
        Krum score per update (lower is more trusted).
    """
    U = np.stack([np.asarray(u, dtype=float) for u in updates])
    n = U.shape[0]
    f = int(n_byzantine)
    if f < 0:
        raise ValueError("n_byzantine must be >= 0")
    if m is None:
        m = max(1, n - f)
    m = int(min(max(1, m), n))

    dist = _pairwise_sq_dist(U)
    scores = _krum_scores(dist, f)
    # Lowest scores = most central / trusted.
    order = np.argsort(scores, kind="stable")
    selected_idx = sorted(int(i) for i in order[:m])
    aggregate = U[selected_idx].mean(axis=0)
    return aggregate, selected_idx, scores


# --------------------------------------------------------------------------- #
# 5. Bulyan.
# --------------------------------------------------------------------------- #
def bulyan(updates, n_byzantine):
    """Bulyan aggregation -- stronger guarantees than Multi-Krum alone.

    Two stages:

    1. *Selection*: run Multi-Krum-style selection repeatedly, each round
       picking the single lowest-Krum-score update and removing it from the
       pool, until ``theta = n - 2f`` updates have been selected. This yields a
       Byzantine-resilient candidate set.
    2. *Aggregation*: for each coordinate, take the ``theta - 2f`` values
       closest to the coordinate-wise median of the selected set and average
       them (a coordinate-wise trimmed mean over the selection).

    Requires ``n >= 4f + 3``. When ``n`` is too small for that bound, we cannot
    honour the Bulyan guarantee, so we document and fall back to
    ``multi_krum_select``.

    Returns
    -------
    aggregate : np.ndarray
    selected_idx : list of int
        Indices kept by the selection stage (ascending).
    scores : np.ndarray
        Krum score per *original* update (lower is more trusted).
    """
    U = np.stack([np.asarray(u, dtype=float) for u in updates])
    n = U.shape[0]
    f = int(n_byzantine)
    if f < 0:
        raise ValueError("n_byzantine must be >= 0")

    # Krum scores over the full set (for reporting / info, regardless of path).
    full_dist = _pairwise_sq_dist(U)
    full_scores = _krum_scores(full_dist, f)

    if n < 4 * f + 3:
        # Not enough members for the Bulyan bound (n >= 4f+3). Degrade
        # gracefully to Multi-Krum, which only needs n >= 2f+3.
        agg, sel, _ = multi_krum_select(updates, n_byzantine=f)
        return agg, sel, full_scores

    theta = n - 2 * f  # size of the selection set
    remaining = list(range(n))
    selected_idx = []

    # Stage 1: greedily peel off the lowest-Krum-score update theta times,
    # recomputing scores on the shrinking pool each round.
    while len(selected_idx) < theta and remaining:
        sub = U[remaining]
        sub_dist = _pairwise_sq_dist(sub)
        sub_scores = _krum_scores(sub_dist, f)
        best_local = int(np.argmin(sub_scores))
        selected_idx.append(remaining.pop(best_local))

    selected_idx_sorted = sorted(selected_idx)
    S = U[selected_idx_sorted]

    # Stage 2: coordinate-wise trimmed mean over the selection. Keep the
    # beta = theta - 2f values nearest the coordinate median, average them.
    beta = max(1, theta - 2 * f)
    med = np.median(S, axis=0)
    # Distance of each selected update's coordinate to the coordinate median.
    closeness = np.abs(S - med[None, :])
    # For each coordinate, indices of the beta closest values.
    order = np.argsort(closeness, axis=0, kind="stable")
    keep = order[:beta, :]  # shape (beta, d)
    aggregate = np.take_along_axis(S, keep, axis=0).mean(axis=0)

    return aggregate, selected_idx_sorted, full_scores


# --------------------------------------------------------------------------- #
# 6. Dispatcher.
# --------------------------------------------------------------------------- #
_METHODS = ("krum", "trimmed_mean", "median", "bulyan")


def robust_aggregate(updates, method, n_byzantine, max_norm=None, **kw):
    """Dispatch to a robust aggregator, optionally norm-clipping first.

    This is the single entry point the control plane should call to replace its
    ``_select`` heuristic. It optionally clips update norms (cheap defence
    against amplification), runs the chosen aggregator, and returns rich
    ``info`` so the caller can report rejected members and fire
    ``attack_detected``.

    Parameters
    ----------
    updates : sequence of np.ndarray
    method : str
        One of ``'krum'``, ``'trimmed_mean'``, ``'median'``, ``'bulyan'``.
    n_byzantine : int
        Assumed number of Byzantine members ``f``.
    max_norm : float, optional
        If given, every update is L2-clipped to this norm before aggregation.
    **kw :
        Method-specific options:
          * ``m``          -> Multi-Krum selection size.
          * ``trim_ratio`` -> trimmed-mean tail fraction (default ``f / n``,
            bumped to at least cover ``f`` per side).

    Returns
    -------
    aggregate : np.ndarray
    info : dict
        Keys:
          * ``method``          -- the method used (may differ if Bulyan fell
            back to Krum; see ``fellback``).
          * ``n``, ``n_byzantine``
          * ``selected``        -- list[int] kept member indices.
          * ``rejected``        -- list[int] dropped member indices.
          * ``scores``          -- list[float] per-update Krum score, or None
            for the median/trimmed-mean paths that are not score-based.
          * ``clipped``         -- bool, whether norm clipping was applied.
          * ``max_norm``        -- the clip value (or None).
          * ``attack_detected`` -- bool, True iff any member was rejected.
          * ``fellback``        -- bool, True iff Bulyan degraded to Krum.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")

    raw = [np.asarray(u, dtype=float) for u in updates]
    n = len(raw)
    f = int(n_byzantine)

    clipped = max_norm is not None
    work = norm_clip(raw, max_norm) if clipped else raw

    info = {
        "method": method,
        "n": n,
        "n_byzantine": f,
        "selected": list(range(n)),
        "rejected": [],
        "scores": None,
        "clipped": clipped,
        "max_norm": max_norm,
        "attack_detected": False,
        "fellback": False,
    }

    if method == "krum":
        agg, selected, scores = multi_krum_select(work, f, m=kw.get("m"))
        info["selected"] = selected
        info["scores"] = [float(s) for s in scores]

    elif method == "bulyan":
        agg, selected, scores = bulyan(work, f)
        info["selected"] = selected
        info["scores"] = [float(s) for s in scores]
        # bulyan() degrades to Krum when n < 4f+3.
        info["fellback"] = n < 4 * f + 3
        if info["fellback"]:
            info["method"] = "krum"

    elif method == "trimmed_mean":
        trim_ratio = kw.get("trim_ratio")
        if trim_ratio is None:
            # Ensure at least f values trimmed per side when possible.
            trim_ratio = (f / n) if n > 0 else 0.0
            if n > 0 and int(np.floor(trim_ratio * n)) < f:
                trim_ratio = min(0.49, (f + 0.0) / n)
        agg = trimmed_mean(work, trim_ratio)
        info["trim_ratio"] = float(trim_ratio)
        # Report which members landed in a trimmed tail on >= half the
        # coordinates, so the caller still gets a "rejected" signal.
        info["selected"], info["rejected"] = _trim_membership(work, trim_ratio)
        info["attack_detected"] = len(info["rejected"]) > 0
        return agg, info

    else:  # median
        agg = coordinate_median(work)
        # Median keeps all members; nothing is formally rejected.
        return agg, info

    # Score-based paths (krum / bulyan): derive rejected + detection flag.
    info["rejected"] = sorted(set(range(n)) - set(info["selected"]))
    info["attack_detected"] = len(info["rejected"]) > 0
    return agg, info


def _trim_membership(updates, trim_ratio):
    """Heuristic membership report for trimmed_mean.

    A member is "rejected" if its value was trimmed (landed in a tail) on a
    majority of coordinates. This is advisory only -- trimmed_mean is
    coordinate-wise and does not select whole members -- but it gives the
    control plane a usable rejected-member signal.
    """
    U = np.stack([np.asarray(u, dtype=float) for u in updates])
    n, d = U.shape[0], U.reshape(U.shape[0], -1).shape[1]
    flat = U.reshape(n, -1)
    k = int(np.floor(trim_ratio * n))
    if k == 0 or n == 0:
        return list(range(n)), []
    trimmed_count = np.zeros(n, dtype=int)
    order = np.argsort(flat, axis=0, kind="stable")  # (n, d)
    low = order[:k, :].reshape(-1)
    high = order[n - k :, :].reshape(-1)
    for idx in np.concatenate([low, high]):
        trimmed_count[idx] += 1
    rejected = sorted(int(i) for i in range(n) if trimmed_count[i] > d // 2)
    selected = sorted(set(range(n)) - set(rejected))
    return selected, rejected
