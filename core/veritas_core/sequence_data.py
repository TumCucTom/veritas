"""Synthetic per-account TRANSACTION SEQUENCES for temporal fraud detection.

Why sequences (and not the per-transaction tabular data in `data.py`)
---------------------------------------------------------------------
Mule / layering fraud is a TEMPORAL typology: the signal is not in any single
transaction but in the *order and rhythm* of an account's transactions over
time. The canonical layering signature here is

    a QUIET period  ->  a sudden BURST of rapid, high-value, high-fan-out
    transactions (the layering / cash-out phase)

A "bag of transactions" model that ignores order sees the same multiset of
feature vectors whether the burst comes first, last, or is scattered — so it is
blind to the signature. A recurrent model that consumes the sequence in time
order can pick it up. The companion test exploits exactly this: shuffling the
timestep order destroys the signature and recall drops.

Shape / API
-----------
`make_sequence_data(n, fraud_rate, T=8, feat=FEATURE_DIM, seed)` returns
`X` of shape (n, T, feat) and integer labels `y` of shape (n,). Each row of X
is one account's T consecutive transaction feature vectors. The feature columns
reuse the conventions from `data.py` (amount[0], ..., velocity[6], fanout[7],
recency[8], campaignSig[-1]) so the sequence model sits alongside the tabular /
graph models. Data is i.i.d. across rows, hence trivially partitionable across
banks (just slice rows) — the same row-slice partitioning the other Veritas
datasets use.
"""
import numpy as np

from .data import FEATURE_DIM


def make_sequence_data(n, fraud_rate, T=8, feat=FEATURE_DIM, seed=0):
    """Generate `n` account transaction sequences of length `T`.

    Returns
    -------
    X : np.ndarray, shape (n, T, feat), float64
        Per-account ordered transaction feature vectors.
    y : np.ndarray, shape (n,), int64
        1 = mule/fraud account (carries the temporal layering signature),
        0 = legitimate.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(np.int64)

    # Baseline "normal" activity: low-amplitude noise around zero for every
    # account and every timestep.
    X = rng.normal(0.0, 0.4, (n, T, feat)).astype(np.float64)

    # A weak per-transaction amplitude bump on the fraud-axis features so that a
    # tabular model gets *some* (but clearly weaker) signal — the temporal model
    # should beat it by using order. Applied to amount[0], velocity[6],
    # fanout[7] uniformly across time.
    amount, velocity, fanout, recency = 0, 6, 7, 8
    X[y == 1, :, amount] += 0.20
    X[y == 1, :, velocity] += 0.20
    X[y == 1, :, fanout] += 0.20

    # Temporal layering signature for fraud accounts: a QUIET prefix followed by
    # a BURST. The burst onset varies per account so the model must read the
    # whole sequence, not a fixed position.
    fraud_idx = np.where(y == 1)[0]
    for i in fraud_idx:
        # burst occupies roughly the back third-to-half of the sequence,
        # starting at a randomised onset in [T//2, T-1].
        onset = int(rng.integers(max(1, T // 2), T))
        # Quiet phase: suppress activity (dormant account).
        X[i, :onset, [amount, velocity, fanout]] -= 0.5
        # Recency rises through the quiet phase (account looks increasingly
        # dormant), giving a ramp the RNN can latch onto before the burst.
        X[i, :onset, recency] += np.linspace(0.0, 1.0, onset)
        # Burst phase: rapid high-value high-fan-out transactions.
        burst = np.arange(onset, T)
        X[i, burst, amount] += 1.6
        X[i, burst, velocity] += 1.6
        X[i, burst, fanout] += 1.6
        X[i, burst, recency] -= 1.0  # transactions now very recent / rapid

    return X, y
