"""REAL credit-card fraud data loader + NON-IID bank partition for Veritas.

Everything else in the Veritas experiments so far is SYNTHETIC. This module
swaps in the REAL, public ULB credit-card fraud dataset (Dal Pozzolo et al.,
ULB / Worldline) — 284,807 genuine European card transactions over two days,
492 of them fraudulent (0.173%). Features are anonymised: 28 PCA components
``V1..V28`` plus ``Time`` and ``Amount``; the label is ``Class`` (1 = fraud).

It exposes two things:

    * :func:`load_creditcard`   — read/standardise the CSV into ``(X, y)``;
    * :func:`partition_noniid`  — split rows across N banks NON-IID so some
      banks are fraud-rich and others fraud-poor (what makes federation matter).

Pure numpy + stdlib (+ urllib for the optional one-time download). The CSV path
is gitignored and is NEVER committed.

Feature / standardisation choices (documented)
----------------------------------------------
Feature matrix columns, in order:

    V1 .. V28                         — the 28 PCA components, used as-is then
                                        z-scored (mean 0, unit variance) per col.
    log_amount = log1p(Amount)        — Amount is heavy-tailed and non-negative;
                                        log1p compresses it, then z-scored.
    time_norm  = Time / max(Time)     — a NORMALISED version of Time in [0, 1].
                                        (We KEEP Time as a normalised column
                                        rather than drop it; it carries weak
                                        diurnal signal and costs one feature.)

So the model sees 30 features (28 PCA + log-amount + normalised-time), all on a
comparable scale. z-scoring uses the GLOBAL train statistics computed on the
whole loaded matrix; this is a benign leak for a federated-vs-siloed *relative*
comparison (both arms see identical standardisation) and keeps the loader
self-contained. ``standardize=False`` returns raw features (still log/Time
transformed) for callers that want to standardise their own split.

Non-IID partition scheme (documented)
-------------------------------------
``partition_noniid`` makes federation matter by concentrating the rare frauds
UNEVENLY across banks:

    1. Negatives (legit rows) are shuffled and split into N roughly-equal,
       contiguous shards — every bank gets a comparable legit base.
    2. Positives (frauds) are shuffled then dealt to banks with a
       MONOTONICALLY-SKEWED probability vector. With
       ``concentration="skewed"`` bank i's share of frauds is proportional to a
       geometric weight ``r**i`` (r<1): bank 0 is fraud-RICH, the last bank is
       fraud-POOR (often only a handful of frauds, sometimes zero). The Dirichlet
       option (``concentration="dirichlet"``) draws the share vector from a
       Dirichlet(alpha) for a softer skew.

The result: a fraud-poor bank cannot learn a good fraud model from its own data
alone (too few / no positives), but it CAN benefit from a federated model that
pools the fraud signal observed at the fraud-rich banks — without any bank
sharing raw rows. Every row lands in exactly one bank (a true partition).
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np

# Public mirror of the ULB credit-card fraud CSV (gitignored target path).
_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/David-Egea/"
    "Creditcard-fraud-detection/resolve/main/creditcard.csv"
)
_DEFAULT_PATH = "core/data/creditcard.csv"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _ensure_csv(path: str, download: bool) -> str:
    """Return a readable CSV path, downloading once if missing (never commits)."""
    if os.path.exists(path):
        return path
    # Be tolerant of being called from core/ with the "core/data/..." default.
    alt = path
    if path.startswith("core/") and not os.path.exists(path):
        alt = path[len("core/"):]
        if os.path.exists(alt):
            return alt
    if not download:
        raise FileNotFoundError(
            f"creditcard.csv not found at {path!r} and download=False")
    target = path
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    urllib.request.urlretrieve(_DOWNLOAD_URL, target)
    return target


def load_creditcard(path: str = _DEFAULT_PATH, download: bool = True,
                    standardize: bool = True):
    """Load the REAL ULB credit-card fraud CSV into ``(X, y)``.

    Features (30 cols, in order): ``V1..V28``, ``log1p(Amount)``,
    ``Time/max(Time)``. With ``standardize=True`` every column is z-scored to
    mean 0 / unit variance using the global statistics of the loaded matrix.
    ``y`` is the integer ``Class`` label in ``{0, 1}``.

    If the CSV is absent and ``download=True`` it is fetched once to ``path``
    (which is gitignored) and cached. Pure numpy parsing.
    """
    csv_path = _ensure_csv(path, download)

    # Header: Time,V1..V28,Amount,Class. Some columns (notably Class) are written
    # QUOTED (e.g. "0"); a quote-stripping converter makes every field parse as a
    # plain float. Parse with numpy, skipping the header row.
    def _unquote(b):
        s = b.decode() if isinstance(b, (bytes, bytearray)) else b
        return float(s.strip().strip('"').strip("'") or "nan")

    ncols = 31  # Time + V1..V28 + Amount + Class
    raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=np.float64,
                        converters={i: _unquote for i in range(ncols)})
    if raw.ndim == 1:  # single row edge-case
        raw = raw.reshape(1, -1)

    time = raw[:, 0]
    v = raw[:, 1:29]           # V1..V28
    amount = raw[:, 29]
    y = raw[:, 30].astype(np.int64)

    log_amount = np.log1p(np.clip(amount, 0.0, None))
    tmax = time.max()
    time_norm = time / tmax if tmax > 0 else np.zeros_like(time)

    X = np.column_stack([v, log_amount, time_norm]).astype(np.float64)

    if standardize:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd

    return X, y


# ---------------------------------------------------------------------------
# Non-IID partition
# ---------------------------------------------------------------------------
def partition_noniid(X, y, n_banks, seed: int = 0, concentration: str = "skewed"):
    """Partition ``(X, y)`` across ``n_banks`` banks NON-IID by fraud rate.

    Negatives are split into N roughly-equal shards; positives (frauds) are
    dealt with a monotonically-skewed share vector so bank 0 is fraud-rich and
    the last bank fraud-poor. Returns a list of ``(X_i, y_i)`` tuples that
    together cover every row exactly once. See module docstring for the scheme.

    ``concentration``:
        "skewed"     — geometric share weights ``r**i`` (default; strong skew).
        "dirichlet"  — share weights ~ Dirichlet(alpha) (softer, random skew).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    # --- negatives: roughly-equal contiguous shards --------------------------
    neg_shards = np.array_split(neg_idx, n_banks)

    # --- positives: skewed share vector -------------------------------------
    if concentration == "dirichlet":
        alpha = 0.3  # <1 => peaky / uneven shares
        shares = rng.dirichlet(np.full(n_banks, alpha))
    else:  # "skewed" geometric default
        r = 0.45
        shares = np.array([r ** i for i in range(n_banks)], dtype=np.float64)
        shares /= shares.sum()

    # Deal each fraud row to a bank sampled from the share vector. This
    # guarantees every positive lands in exactly one bank while honouring the
    # skew; small banks may receive zero frauds (the point of the experiment).
    pos_assign = rng.choice(n_banks, size=pos_idx.size, p=shares)

    banks = []
    for b in range(n_banks):
        b_pos = pos_idx[pos_assign == b]
        b_neg = neg_shards[b]
        idx = np.concatenate([b_neg, b_pos])
        rng.shuffle(idx)
        banks.append((X[idx], y[idx]))
    return banks
