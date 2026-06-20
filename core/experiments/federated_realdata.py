"""REAL-DATA EXPERIMENT: federated beats siloed on the ULB credit-card fraud set.

Everything else in the Veritas experiments is synthetic. This one runs the SAME
federation machinery on the REAL, public ULB credit-card fraud dataset
(284,807 real EU card transactions, 492 frauds = 0.173%) and MEASURES the
federated-vs-siloed lift.

Run from core/ with the venv active:

    python -m experiments.federated_realdata

What it does
------------
1. Loads the REAL CSV (``veritas_core.realdata.load_creditcard``): 30 features
   (V1..V28 PCA + log1p(Amount) + normalised Time), z-scored; label = Class.
2. Subsamples to ~80k rows (keeping ALL frauds) for speed — documented below.
3. Partitions NON-IID across N banks so some banks are fraud-rich and some
   fraud-poor (``realdata.partition_noniid``). A held-out test set is carved out
   first (stratified) and is shared, so every arm is scored on the same rows.
4. Trains, for logistic / MLP / stacked-ensemble, two arms:
       SILOED    — each bank trains only on its own rows; we score the *mean*
                   per-bank model AUPRC (what a bank gets alone) AND the
                   fraud-poor bank specifically.
       FEDERATED — one model trained across all banks via the existing FL
                   primitives (fedavg on the flat weight vector; the ensemble
                   federates its bases + meta-learner).
5. Reports IMBALANCE-APPROPRIATE metrics — raw recall@0.5 is meaningless at
   0.173%, so we report:
       AUPRC          = area under precision-recall curve (average precision).
       recall@budget  = fraction of all frauds caught inside a fixed alert
                        budget (top-0.5% highest-scored test rows) — the metric
                        a fraud ops team actually lives by.
   and prints a clear siloed-vs-federated table with the lift.

Subsampling note: the experiment subsamples negatives to keep runtime modest on
pure-numpy models while KEEPING ALL 492 frauds (so the rare-class signal is
intact). Set ``SUBSAMPLE=None`` to use the full 284,807 rows.
"""
from __future__ import annotations

import numpy as np

from veritas_core import realdata
from veritas_core import model as logistic
from veritas_core import mlp
from veritas_core.ensemble import (
    StackedEnsemble, LogisticAdapter, MLPAdapter, GBDTAdapter,
)

# ---- experiment configuration ---------------------------------------------
N_BANKS = 5
SUBSAMPLE = 80_000      # total rows for the experiment (keeps ALL frauds); None = full
TEST_FRAC = 0.30
ALERT_BUDGET = 0.005    # top-0.5% scored rows = the alert budget
SEED = 0


# ---- imbalance-appropriate metrics ----------------------------------------
def auprc(scores, y):
    """Average precision (area under the precision-recall curve), pure numpy."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y).astype(np.int64)
    P = int((y == 1).sum())
    if P == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.clip(tp + fp, 1, None)
    recall = tp / P
    # AP = sum over thresholds of (recall[k] - recall[k-1]) * precision[k]
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - rec_prev) * precision))


def recall_at_budget(scores, y, budget=ALERT_BUDGET):
    """Recall among the top-``budget`` fraction of highest-scored rows."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y).astype(np.int64)
    P = int((y == 1).sum())
    if P == 0:
        return 0.0
    n = len(scores)
    k = max(1, int(round(n * budget)))
    top = np.argsort(-scores, kind="mergesort")[:k]
    return float((y[top] == 1).sum() / P)


# ---- data prep -------------------------------------------------------------
def _stratified_split(X, y, test_frac, rng):
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    n_pos_te = max(1, int(round(len(pos) * test_frac)))
    n_neg_te = max(1, int(round(len(neg) * test_frac)))
    te = np.concatenate([pos[:n_pos_te], neg[:n_neg_te]])
    tr = np.concatenate([pos[n_pos_te:], neg[n_neg_te:]])
    rng.shuffle(te); rng.shuffle(tr)
    return X[tr], y[tr], X[te], y[te]


def _subsample(X, y, n_total, rng):
    """Keep ALL positives; subsample negatives to hit ~n_total rows."""
    if n_total is None or n_total >= len(y):
        return X, y
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    n_neg = max(1, n_total - len(pos))
    neg_keep = rng.choice(neg, size=min(n_neg, len(neg)), replace=False)
    idx = np.concatenate([pos, neg_keep]); rng.shuffle(idx)
    return X[idx], y[idx]


# ---- model arms ------------------------------------------------------------
def _logistic_siloed(banks):
    return [logistic.train_local(logistic.init_weights(b[0].shape[1]),
                                 b[0], b[1], epochs=60, lr=0.3)
            for b in banks]


def _logistic_federated(banks, rounds=8):
    w = logistic.init_weights(banks[0][0].shape[1])
    from veritas_core.aggregation import fedavg
    for _ in range(rounds):
        updates = [logistic.train_local(w, b[0], b[1], epochs=15, lr=0.3)
                   for b in banks]
        w = fedavg(updates)
    return w


def _mlp_siloed(banks):
    out = []
    for b in banks:
        d = b[0].shape[1]
        w = mlp.init_weights(in_dim=d, seed=SEED)
        w = mlp.train_local(w, b[0], b[1], epochs=60, lr=0.1, in_dim=d)
        out.append((w, d))
    return out


def _mlp_federated(banks, rounds=8):
    from veritas_core.aggregation import fedavg
    d = banks[0][0].shape[1]
    w = mlp.init_weights(in_dim=d, seed=SEED)
    for _ in range(rounds):
        updates = [mlp.train_local(w, b[0], b[1], epochs=12, lr=0.1, in_dim=d)
                   for b in banks]
        w = fedavg(updates)
    return w, d


def _bank_views(banks):
    return [{"tab": b[0], "y": b[1]} for b in banks]


def _ensemble_bases(seed=SEED):
    return [
        LogisticAdapter(seed=seed, rounds=8, local_epochs=15, lr=0.3),
        MLPAdapter(seed=seed, rounds=8, local_epochs=12, lr=0.1),
        GBDTAdapter(n_trees=25, max_depth=3, n_bins=16, lr=0.3),
    ]


# ---- evaluation helpers ----------------------------------------------------
def _eval(scores, y):
    return auprc(scores, y), recall_at_budget(scores, y)


def _fmt(v):
    return f"{v:6.4f}"


def run():
    rng = np.random.default_rng(SEED)
    print("Loading REAL ULB credit-card fraud data ...")
    X, y = realdata.load_creditcard(standardize=True)
    print(f"  full dataset: {len(y):,} rows, {int(y.sum())} frauds "
          f"({100*y.mean():.3f}% fraud rate), {X.shape[1]} features")

    X, y = _subsample(X, y, SUBSAMPLE, rng)
    Xtr, ytr, Xte, yte = _stratified_split(X, y, TEST_FRAC, rng)
    print(f"  experiment: {len(y):,} rows ({int(y.sum())} frauds); "
          f"train {len(ytr):,} / test {len(yte):,} (frauds in test: {int(yte.sum())})")

    banks = realdata.partition_noniid(Xtr, ytr, N_BANKS, seed=SEED,
                                      concentration="skewed")
    fraud_rates = [100 * b[1].mean() for b in banks]
    fraud_counts = [int(b[1].sum()) for b in banks]
    print("\nNON-IID partition (fraud rate per bank, bank 0 = richest):")
    for i, (r, c, b) in enumerate(zip(fraud_rates, fraud_counts, banks)):
        print(f"  bank {i}: {len(b[1]):>7,} rows, {c:>3} frauds, {r:5.3f}% fraud")
    poor_bank = int(np.argmin(fraud_counts))
    print(f"  -> fraud-poorest bank = bank {poor_bank} "
          f"({fraud_counts[poor_bank]} frauds)")

    rows = []  # (model, siloed_auprc, fed_auprc, siloed_rab, fed_rab)

    # ----- LOGISTIC ---------------------------------------------------------
    sil_w = _logistic_siloed(banks)
    sil_scores = [logistic.predict_proba(w, Xte) for w in sil_w]
    sil_metrics = [_eval(s, yte) for s in sil_scores]
    sil_auprc = np.mean([m[0] for m in sil_metrics])
    sil_rab = np.mean([m[1] for m in sil_metrics])
    poor_auprc = sil_metrics[poor_bank][0]
    fed_w = _logistic_federated(banks)
    fa, fr = _eval(logistic.predict_proba(fed_w, Xte), yte)
    rows.append(("logistic", sil_auprc, fa, sil_rab, fr, poor_auprc))

    # ----- MLP --------------------------------------------------------------
    sil = _mlp_siloed(banks)
    sil_scores = [mlp.predict_proba(w, Xte, in_dim=d) for (w, d) in sil]
    sil_metrics = [_eval(s, yte) for s in sil_scores]
    sil_auprc = np.mean([m[0] for m in sil_metrics])
    sil_rab = np.mean([m[1] for m in sil_metrics])
    poor_auprc_mlp = sil_metrics[poor_bank][0]
    fw, fd = _mlp_federated(banks)
    fa, fr = _eval(mlp.predict_proba(fw, Xte, in_dim=fd), yte)
    rows.append(("mlp", sil_auprc, fa, sil_rab, fr, poor_auprc_mlp))

    # ----- STACKED ENSEMBLE -------------------------------------------------
    test_view = [{"tab": Xte, "y": yte}]
    # siloed: train one ensemble per bank on that bank's data only
    sil_metrics = []
    for b in banks:
        ens = StackedEnsemble(_ensemble_bases(), meta_rounds=6)
        ens.fed_train([{"tab": b[0], "y": b[1]}], meta_frac=0.4, seed=SEED)
        s = ens.predict_proba(test_view)
        sil_metrics.append(_eval(s, yte))
    sil_auprc = np.mean([m[0] for m in sil_metrics])
    sil_rab = np.mean([m[1] for m in sil_metrics])
    poor_auprc_ens = sil_metrics[poor_bank][0]
    # federated: one ensemble across all banks
    ens = StackedEnsemble(_ensemble_bases(), meta_rounds=6)
    ens.fed_train(_bank_views(banks), meta_frac=0.4, seed=SEED)
    fa, fr = _eval(ens.predict_proba(test_view), yte)
    rows.append(("ensemble", sil_auprc, fa, sil_rab, fr, poor_auprc_ens))

    # ----- TABLE ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("REAL-DATA federated-vs-siloed (ULB credit-card fraud, 0.173% positives)")
    print("metrics: AUPRC = average precision ; R@budget = recall in top-0.5% alerts")
    print("=" * 78)
    print(f"{'model':<10} | {'AUPRC siloed':>12} {'AUPRC fed':>10} {'lift':>7} "
          f"| {'R@bud silo':>11} {'R@bud fed':>10} {'lift':>7}")
    print("-" * 78)
    for (name, sa, fa, sr, fr, _poor) in rows:
        la = fa - sa
        lr = fr - sr
        print(f"{name:<10} | {_fmt(sa):>12} {_fmt(fa):>10} {la:+7.4f} "
              f"| {_fmt(sr):>11} {_fmt(fr):>10} {lr:+7.4f}")
    print("-" * 78)
    print(f"fraud-poorest bank (bank {poor_bank}) AUPRC alone vs federated:")
    for (name, sa, fa, sr, fr, poor) in rows:
        print(f"  {name:<10}: poor-bank-siloed AUPRC {_fmt(poor)} "
              f"-> federated {_fmt(fa)}  (lift {fa - poor:+.4f})")
    print("=" * 78)
    return rows


if __name__ == "__main__":
    run()
