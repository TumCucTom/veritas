"""OFFLINE viz generator for the Veritas STACKED FEDERATED ENSEMBLE.

Runs the project's REAL pure-numpy stacked ensemble (``veritas_core.ensemble``)
on the REAL heterogeneous multi-view fraud dataset from the headline experiment
(``experiments.federated_ensemble``) and exports a precomputed, compact JSON
artifact to ``web/public/viz/ensemble.json``. The website animates it: each base
model "votes" with its per-sample score, the federated logistic META-LEARNER
stacks those votes (edge weight = the meta-learner's learned coefficient for that
base), and the ensemble verdict resolves — demonstrably beating any single model.

Everything is deterministic (seed 1234) and re-runnable.

What is REAL here
-----------------
  * Base models: logistic + MLP + GBDT + GRU sequence + categorical embeddings,
    each FEDERATED-trained across banks via ``ensemble.default_base_learners`` /
    ``ensemble.fed_train_ensemble`` (NN bases fedavg the flat weight vector, GBDT
    its histogram protocol).
  * The META-LEARNER is the ensemble's real federated logistic regression over
    the five base probabilities; we read its learned COEFFICIENTS straight off
    ``ensemble.meta_w`` (one per base model) and surface them as edge "weights"
    (how much the stack trusts each base).
  * Per-model standalone metric (recall + AUC) and the ENSEMBLE's metric are
    computed on a held-out test set carrying ALL four fraud typologies — the
    same numbers the experiment's CLI prints.
  * Sample transactions carry every base model's REAL score, the REAL
    meta-combined final score, and the true label.

Run:  python tools/gen_ensemble_viz.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# Make veritas_core + experiments importable when run from repo root.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "core"))

from veritas_core import ensemble as ens  # noqa: E402
from veritas_core import model as logistic  # noqa: E402
from experiments import federated_ensemble as fe  # noqa: E402

SEED = 1234
OUT_DIR = os.path.join(REPO, "web", "public", "viz")
N_SAMPLES = 6  # transactions surfaced for the "voting" animation

# Display names + the headline sub-pattern each base family owns.
DISPLAY = {
    "logistic": "Logistic",
    "mlp": "MLP",
    "gbdt": "Federated GBDT",
    "sequence": "GRU sequence",
    "embeddings": "Embeddings",
}


def _r(x, nd=4):
    return round(float(x), nd)


def build():
    """Train the real stacked ensemble and assemble the viz payload."""
    # REAL federated multi-view data (4 banks, each blind to a different typology)
    train_views = fe.make_bank_views(n_banks=4, n_per_bank=1500, seed=SEED)
    test_views = fe.make_test_views(seed=SEED + 999)
    y_test = np.concatenate([v["y"] for v in test_views])

    # Build + federated-train the full five-base stacked ensemble.
    base_learners = ens.default_base_learners(seed=0)
    ensemble = ens.fed_train_ensemble(train_views, base_learners,
                                      meta_frac=0.4, seed=SEED)

    # Per-base scores on the held-out test set (each base already federated).
    base_keys = [b.name for b in ensemble.bases]
    base_scores = np.column_stack([
        ensemble.base_predict_proba(i, test_views)
        for i in range(len(ensemble.bases))
    ])  # (n_test, n_base)
    final_scores = ensemble.predict_proba(test_views)  # (n_test,)

    # Meta-learner learned coefficients: ensemble.meta_w is a flat logistic
    # weight vector over the base-prob features (one coef per base + a bias).
    # logistic.init_weights(n) is length n+1, so the first n entries are the
    # per-base coefficients (the bias is the trailing entry) — exactly how much
    # the stack trusts each base model.
    meta_w = np.asarray(ensemble.meta_w, dtype=np.float64)
    coefs = meta_w[:len(ensemble.bases)]

    # Per-base standalone metric (recall + AUC at thr 0.5).
    models = []
    for i, b in enumerate(ensemble.bases):
        s = base_scores[:, i]
        pred = s > 0.5
        recall = float((pred & (y_test == 1)).sum() / max(1, (y_test == 1).sum()))
        auc = ens.auc(s, y_test)
        models.append({
            "key": b.name,
            "name": DISPLAY.get(b.name, b.name),
            "metric": _r(recall),
            "metricLabel": "recall",
            "weight": _r(float(coefs[i])),
            "_auc": _r(auc),
        })

    # Ensemble metric.
    e_pred = final_scores > 0.5
    e_recall = float((e_pred & (y_test == 1)).sum() / max(1, (y_test == 1).sum()))
    e_auc = ens.auc(final_scores, y_test)

    # ~6 sample transactions for the voting animation. Pick a legible spread:
    # half genuine fraud (label 1) and half legit (label 0), choosing rows where
    # the bases visibly DISAGREE so the meta-learner's combination reads clearly.
    samples = _pick_samples(base_scores, final_scores, y_test, base_keys)

    note = ("Real five-model stacked federated ensemble (logistic + MLP + "
            "federated GBDT + GRU sequence + categorical embeddings) combined "
            "by a federated logistic meta-learner over the base probabilities. "
            "Each base owns a different fraud typology and is blind to the "
            "others; 'weight' is the meta-learner's learned coefficient for that "
            "base (how much the stack trusts it). Metrics are recall at "
            "threshold 0.5 on a held-out test set carrying all four typologies; "
            "the ensemble beats every single base because it unions their "
            "coverage. All numbers are real model outputs, deterministic at "
            "seed 1234.")

    out = {
        "models": [
            {"key": m["key"], "name": m["name"], "metric": m["metric"],
             "metricLabel": m["metricLabel"], "weight": m["weight"]}
            for m in models
        ],
        "ensemble": {"metric": _r(e_recall), "metricLabel": "recall"},
        "samples": samples,
        "meta": {"note": note, "metricName": "recall"},
    }
    summary = {
        "models": models,
        "ensemble_recall": e_recall,
        "ensemble_auc": e_auc,
        "n_test": int(len(y_test)),
        "n_test_fraud": int((y_test == 1).sum()),
        "best_single": max(models, key=lambda m: m["metric"]),
    }
    return out, summary


def _pick_samples(base_scores, final_scores, y, base_keys):
    """Choose N_SAMPLES legible transactions (balanced labels, high disagreement).

    Disagreement = spread of base scores on a row; high-spread rows make the
    "models vote, meta combines" story read clearly. Deterministic tie-break by
    row index.
    """
    spread = base_scores.max(axis=1) - base_scores.min(axis=1)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_pos = N_SAMPLES // 2
    n_neg = N_SAMPLES - n_pos
    # most-disagreed rows per class (stable order)
    pos_pick = pos[np.argsort(-spread[pos], kind="mergesort")[:n_pos]]
    neg_pick = neg[np.argsort(-spread[neg], kind="mergesort")[:n_neg]]
    # interleave fraud/legit so the animation alternates verdicts
    order = []
    for a, b in zip(pos_pick, neg_pick):
        order.extend([a, b])
    order.extend(pos_pick[len(neg_pick):])
    order.extend(neg_pick[len(pos_pick):])

    samples = []
    for n, idx in enumerate(order):
        samples.append({
            "id": int(n),
            "label": int(y[idx]),
            "baseScores": {k: _r(float(base_scores[idx, j]))
                           for j, k in enumerate(base_keys)},
            "finalScore": _r(float(final_scores[idx])),
        })
    return samples


def validate(d):
    assert set(d) == {"models", "ensemble", "samples", "meta"}, set(d)
    keys = set()
    for m in d["models"]:
        assert set(m) == {"key", "name", "metric", "metricLabel", "weight"}, set(m)
        assert 0.0 <= m["metric"] <= 1.0
        assert isinstance(m["name"], str) and m["name"]
        keys.add(m["key"])
    assert len(keys) == len(d["models"])  # unique keys
    assert set(d["ensemble"]) == {"metric", "metricLabel"}
    assert 0.0 <= d["ensemble"]["metric"] <= 1.0
    assert d["meta"]["metricName"] in ("AUC", "recall")
    assert isinstance(d["meta"]["note"], str) and d["meta"]["note"]
    assert len(d["samples"]) >= 4
    for s in d["samples"]:
        assert set(s) == {"id", "label", "baseScores", "finalScore"}, set(s)
        assert s["label"] in (0, 1)
        assert 0.0 <= s["finalScore"] <= 1.0
        assert set(s["baseScores"].keys()) == keys
        assert all(0.0 <= v <= 1.0 for v in s["baseScores"].values())
    # the whole point: ensemble should beat the best single base
    best_single = max(m["metric"] for m in d["models"])
    assert d["ensemble"]["metric"] >= best_single, \
        f"ensemble {d['ensemble']['metric']} did not beat best single {best_single}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(SEED)

    print("Training REAL stacked federated ensemble (5 base models) ...")
    out, summary = build()

    path = os.path.join(OUT_DIR, "ensemble.json")
    with open(path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))

    with open(path) as fh:
        validate(json.load(fh))

    size_kb = os.path.getsize(path) / 1024.0

    print("\n==================== SUMMARY ====================")
    print(f"ensemble.json   {size_kb:7.2f} KB  {path}")
    print(f"  test={summary['n_test']} ({summary['n_test_fraud']} fraud)  "
          f"samples={len(out['samples'])}")
    print("-" * 56)
    print(f"  {'model':<16}{'recall':>8}{'AUC':>8}{'meta-weight':>13}")
    print("-" * 56)
    for m in summary["models"]:
        print(f"  {m['name']:<16}{m['metric']:>8.3f}{m['_auc']:>8.3f}"
              f"{m['weight']:>13.3f}")
    print("-" * 56)
    print(f"  {'ENSEMBLE':<16}{summary['ensemble_recall']:>8.3f}"
          f"{summary['ensemble_auc']:>8.3f}")
    best = summary["best_single"]
    print(f"  best single base : {best['name']} (recall {best['metric']:.3f})")
    won = summary["ensemble_recall"] >= best["metric"]
    print("  RESULT: ENSEMBLE WINS" if won
          else "  RESULT: ensemble did NOT beat best single")
    print("=================================================")
    print(f"ensemble.json written, schema-validated, {size_kb:.1f} KB (< 120 KB).")


if __name__ == "__main__":
    main()
