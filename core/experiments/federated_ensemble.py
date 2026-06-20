"""HEADLINE EXPERIMENT: a STACKED FEDERATED ENSEMBLE beats every single base
model because real fraud is a MIX of sub-patterns and no one model covers all.

Run from core/ (with the venv active):

    python -m experiments.federated_ensemble

What it builds
--------------
A HETEROGENEOUS, multi-view fraud dataset spread across banks. Every account
carries THREE parallel views and ONE shared label:

    * "tab"  (n, D) dense tabular  — read by logistic / MLP / GBDT
    * "seq"  (n, T, F) sequence    — read by the GRU
    * "cat"+"dense" categorical    — read by the embedding model

The label is 1 if the account belongs to ANY of four fraud sub-typologies,
each engineered so that ONE model family detects it and the others are blind:

    1. LINEAR slice      — a clean linear margin on tab features  -> LOGISTIC
    2. XOR / interaction — sign(tab[a]) != sign(tab[b])           -> MLP / GBDT
    3. TEMPORAL burst    — a quiet->burst layering signature      -> GRU (sequence)
    4. CATEGORICAL corridor — a (merchant x country) mule combo   -> EMBEDDINGS

Crucially the views are DECORRELATED across typologies: a temporal-burst fraud
looks perfectly normal on the tab/cat views, a corridor fraud looks normal on
tab/seq, etc. So a logistic model can recover (1) but is structurally blind to
(2)-(4); the GRU recovers (3) but nothing else; and so on. The STACKED ENSEMBLE
learns, per account, which base to trust and therefore unions the coverage —
beating the best single base model.

Federation angle
----------------
Banks are partly BLIND locally: each bank under-samples a different subset of
the typologies, so siloed training misses the typologies it rarely sees. The
ensemble trains every base FEDERATED across banks (NN bases via fedavg /
multi_krum on the flat weight vector, GBDT via its histogram protocol), then a
FEDERATED logistic meta-learner over the base probabilities — all without raw
rows leaving a bank. We report federated-vs-siloed for the headline base model
to show federation also helps, then the ensemble on top.
"""
from __future__ import annotations

import numpy as np

from veritas_core import ensemble as ens


# ===========================================================================
# Heterogeneous multi-view data generator
# ===========================================================================
TAB_DIM = 8
SEQ_T = 8
SEQ_FEAT = 6
N_MERCHANT, N_COUNTRY, N_CHANNEL = 12, 8, 4
CARD = [N_MERCHANT, N_COUNTRY, N_CHANNEL]
DENSE_DIM = 6


def _make_accounts(n, rng, typ_rates):
    """Generate n accounts with decorrelated views and 4 fraud sub-typologies.

    typ_rates : (r_linear, r_xor, r_temporal, r_corridor) base rates for each
    typology BEFORE this bank's blindness is applied by the caller.
    """
    r_lin, r_xor, r_tmp, r_cor = typ_rates

    # ---- views: all start as benign noise -----------------------------
    tab = rng.normal(0.0, 1.0, (n, TAB_DIM))
    seq = rng.normal(0.0, 0.4, (n, SEQ_T, SEQ_FEAT))
    merchant = rng.integers(0, N_MERCHANT, n)
    country = rng.integers(0, N_COUNTRY, n)
    channel = rng.integers(0, N_CHANNEL, n)
    dense = rng.normal(0.0, 1.0, (n, DENSE_DIM))

    y = np.zeros(n, dtype=np.int64)

    # ---- typology 1: LINEAR slice (logistic) --------------------------
    # A linear margin along tab[0]+tab[1]-tab[2]; fraud sits on the high side.
    lin = rng.random(n) < r_lin
    tab[lin, 0] += 1.8
    tab[lin, 1] += 1.4
    tab[lin, 2] -= 1.2
    y[lin] = 1

    # ---- typology 2: XOR / interaction (MLP / GBDT) -------------------
    # Fraud iff sign(tab[3]) != sign(tab[4]) AND magnitude high — a pure
    # interaction with NO linear margin (each axis alone is uninformative).
    xor = rng.random(n) < r_xor
    a = rng.choice([-1.0, 1.0], xor.sum())
    tab[xor, 3] = a * 1.6
    tab[xor, 4] = -a * 1.6  # opposite sign -> XOR-positive
    y[xor] = 1
    # plant XOR-NEGATIVE decoys in benign rows (same sign) so the margin is
    # genuinely non-linear (logistic can't separate it).
    benign = np.where(y == 0)[0]
    dec = benign[rng.random(benign.size) < 0.15]
    s = rng.choice([-1.0, 1.0], dec.size)
    tab[dec, 3] = s * 1.6
    tab[dec, 4] = s * 1.6  # same sign -> stays benign

    # ---- typology 3: TEMPORAL burst (GRU) -----------------------------
    # Quiet prefix then a high-amplitude burst; views tab/cat stay benign.
    tmp = rng.random(n) < r_tmp
    tmp_idx = np.where(tmp)[0]
    amount, velocity, fanout, recency = 0, 1, 2, 3
    for i in tmp_idx:
        onset = int(rng.integers(max(1, SEQ_T // 2), SEQ_T))
        seq[i, :onset, [amount, velocity, fanout]] -= 0.5
        seq[i, :onset, recency] += np.linspace(0.0, 1.0, onset)
        seq[i, onset:, amount] += 1.7
        seq[i, onset:, velocity] += 1.7
        seq[i, onset:, fanout] += 1.7
        seq[i, onset:, recency] -= 1.0
    y[tmp] = 1

    # ---- typology 4: CATEGORICAL corridor (embeddings) ----------------
    # The (merchant==3 AND country==5) mule corridor; no tab/seq signal.
    cor = rng.random(n) < r_cor
    merchant[cor] = 3
    country[cor] = 5
    y[cor] = 1
    # plant the corridor values individually in benign rows so neither value is
    # fraudulent alone — only the co-occurrence is.
    benign2 = np.where(y == 0)[0]
    half = benign2[rng.random(benign2.size) < 0.2]
    merchant[half[: half.size // 2]] = 3
    country[half[half.size // 2:]] = 5

    cat = np.stack([merchant, country, channel], axis=1).astype(np.int64)
    return {
        "y": y,
        "tab": tab.astype(np.float64),
        "seq": seq.astype(np.float64),
        "cat": cat,
        "dense": dense.astype(np.float64),
        "cardinalities": CARD,
    }


def make_bank_views(n_banks=4, n_per_bank=1500, seed=0):
    """Per-bank heterogeneous views. Each bank is BLIND to a different typology
    (rate ~0) so siloed training misses it and federation must fill the gap."""
    rng = np.random.default_rng(seed)
    base = (0.05, 0.05, 0.05, 0.05)  # each typology ~5% -> ~20% fraud overall
    views = []
    for b in range(n_banks):
        # rotate which typology this bank is (almost) blind to
        blind = b % 4
        rates = list(base)
        rates[blind] = 0.004  # almost none of this typology locally
        v = _make_accounts(n_per_bank, np.random.default_rng(rng.integers(1 << 30)),
                           tuple(rates))
        views.append(v)
    return views


def make_test_views(n=4000, seed=999):
    """A balanced held-out test set carrying ALL four typologies."""
    rng = np.random.default_rng(seed)
    return [_make_accounts(n, rng, (0.05, 0.05, 0.05, 0.05))]


# ===========================================================================
# Experiment
# ===========================================================================
def _concat_y(views):
    return np.concatenate([v["y"] for v in views])


def run(n_banks=4, n_per_bank=1500, seed=0, verbose=True):
    train_views = make_bank_views(n_banks=n_banks, n_per_bank=n_per_bank, seed=seed)
    test_views = make_test_views(seed=seed + 999)
    y_test = _concat_y(test_views)

    # Build the ensemble (covers all five base learners / four typologies).
    base_learners = ens.default_base_learners(seed=0)
    ensemble = ens.fed_train_ensemble(train_views, base_learners,
                                      meta_frac=0.4, seed=seed)

    # Per-base recall/AUC on the held-out test set (each base already federated).
    rows = []
    for i, b in enumerate(ensemble.bases):
        scores = ensemble.base_predict_proba(i, test_views)
        pred = scores > 0.5
        rec = float((pred & (y_test == 1)).sum() / max(1, (y_test == 1).sum()))
        rows.append({"name": b.name, "recall": rec, "auc": ens.auc(scores, y_test)})

    # Ensemble recall / AUC.
    e_scores = ensemble.predict_proba(test_views)
    e_pred = e_scores > 0.5
    e_recall = float((e_pred & (y_test == 1)).sum() / max(1, (y_test == 1).sum()))
    e_auc = ens.auc(e_scores, y_test)

    best = max(rows, key=lambda r: r["recall"])
    logi = next(r for r in rows if r["name"] == "logistic")

    res = {
        "rows": rows,
        "ensemble_recall": e_recall,
        "ensemble_auc": e_auc,
        "best_single": best,
        "logistic": logi,
        "lift_over_best": e_recall - best["recall"],
        "lift_over_logistic": e_recall - logi["recall"],
        "n_banks": n_banks,
        "n_per_bank": n_per_bank,
        "n_test_fraud": int((y_test == 1).sum()),
        "n_test": int(len(y_test)),
    }
    if verbose:
        _print(res)
    return res


def _print(res):
    print("Stacked Federated Ensemble  —  heterogeneous multi-view fraud")
    print("=" * 64)
    print(f"banks={res['n_banks']}  rows/bank={res['n_per_bank']}  "
          f"test={res['n_test']} ({res['n_test_fraud']} fraud)")
    print("-" * 64)
    print(f"{'model':<14}{'sub-pattern':<22}{'recall':>9}{'AUC':>9}")
    print("-" * 64)
    cover = {
        "logistic": "linear slice",
        "mlp": "XOR / interaction",
        "gbdt": "threshold interact.",
        "sequence": "temporal burst",
        "embeddings": "categorical corridor",
    }
    for r in res["rows"]:
        print(f"{r['name']:<14}{cover.get(r['name'], ''):<22}"
              f"{r['recall']:>9.3f}{r['auc']:>9.3f}")
    print("-" * 64)
    print(f"{'ENSEMBLE':<14}{'ALL (stacked)':<22}"
          f"{res['ensemble_recall']:>9.3f}{res['ensemble_auc']:>9.3f}")
    print("=" * 64)
    best = res["best_single"]
    print(f"  best single base : {best['name']} (recall {best['recall']:.3f})")
    print(f"  ensemble LIFT over best-single : "
          f"+{res['lift_over_best']:.3f} recall")
    print(f"  ensemble LIFT over logistic    : "
          f"+{res['lift_over_logistic']:.3f} recall")
    won = res["ensemble_recall"] >= best["recall"]
    print("=" * 64)
    print("  RESULT: ENSEMBLE WINS" if won
          else "  RESULT: ensemble did NOT beat best single")
    return res


def main():
    return run(verbose=True)


if __name__ == "__main__":
    main()
