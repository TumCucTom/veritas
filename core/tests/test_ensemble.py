import numpy as np

from veritas_core import ensemble as ens
from veritas_core.ensemble import (
    StackedEnsemble, LogisticAdapter, MLPAdapter, GBDTAdapter,
    SequenceAdapter, EmbeddingAdapter, fed_train_ensemble, split_view, auc,
)
from veritas_core import model as logistic
from veritas_core.aggregation import fedavg, multi_krum
from experiments.federated_ensemble import make_bank_views, make_test_views


# --------------------------------------------------------------------------
# Shared fixtures (built once, reused across tests for speed)
# --------------------------------------------------------------------------
def _train_ensemble(seed=0):
    train_views = make_bank_views(n_banks=4, n_per_bank=1200, seed=seed)
    base = ens.default_base_learners(seed=0)
    e = fed_train_ensemble(train_views, base, meta_frac=0.4, seed=seed)
    return e


def _per_base_recall(e, test_views, y_test):
    recs = {}
    for i, b in enumerate(e.bases):
        s = e.base_predict_proba(i, test_views)
        recs[b.name] = float(((s > 0.5) & (y_test == 1)).sum() / (y_test == 1).sum())
    return recs


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_ensemble_beats_best_single_and_logistic():
    """On heterogeneous multi-view data the stacked ensemble recall must be
    >= the best single base model (with margin) and >> logistic baseline."""
    e = _train_ensemble(seed=1)
    test_views = make_test_views(n=3000, seed=4321)
    y_test = test_views[0]["y"]

    base_recs = _per_base_recall(e, test_views, y_test)
    best_single = max(base_recs.values())
    logi = base_recs["logistic"]

    e_pred = e.predict_proba(test_views) > 0.5
    e_recall = float((e_pred & (y_test == 1)).sum() / (y_test == 1).sum())

    # ensemble unions coverage -> at least matches best single (small margin)
    assert e_recall >= best_single - 1e-9
    assert e_recall >= best_single + 0.02
    # and is far above the logistic baseline (logistic only covers 1 of 4 typologies)
    assert e_recall >= logi + 0.2


def test_predict_proba_in_unit_interval():
    e = _train_ensemble(seed=2)
    test_views = make_test_views(n=800, seed=77)
    p = e.predict_proba(test_views)
    assert p.shape == (len(test_views[0]["y"]),)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_meta_learner_federates_consistent_dim_and_aggregators_accept():
    """The meta-learner is a federated logistic over the base probabilities:
    its flat weight vector has a consistent dim across banks and fedavg /
    multi_krum accept a list of them."""
    train_views = make_bank_views(n_banks=4, n_per_bank=1000, seed=3)
    base = ens.default_base_learners(seed=0)
    e = StackedEnsemble(base, meta_rounds=3)

    # split + train bases exactly as fed_train does, then build per-bank
    # meta-learner updates from the held-out meta features.
    splits = [split_view(v, meta_frac=0.4, seed=3 + i) for i, v in enumerate(train_views)]
    base_views = [s[0] for s in splits]
    meta_views = [s[1] for s in splits]
    for b in e.bases:
        b.fed_train(base_views)

    n_feat = len(e.bases)
    w0 = logistic.init_weights(n_feat)
    updates = []
    for v in meta_views:
        mx = np.column_stack([b.predict(v) for b in e.bases])
        w = logistic.train_local(w0, mx, v["y"], epochs=50, lr=0.3)
        updates.append(w)

    expected = n_feat + 1  # logistic packs a bias term
    assert all(u.shape == (expected,) for u in updates)

    avg = fedavg(updates)
    assert avg.shape == (expected,)

    krum, sel = multi_krum(updates, n_byzantine=1, m=2)
    assert krum.shape == (expected,)
    assert len(sel) == 2


def test_full_fed_train_meta_weights_consistent_dim():
    e = _train_ensemble(seed=5)
    assert e.meta_w.shape == (len(e.bases) + 1,)


def test_robust_aggregation_path_runs():
    """multi_krum federation path (robust=True) trains and predicts in range."""
    train_views = make_bank_views(n_banks=4, n_per_bank=900, seed=6)
    base = [LogisticAdapter(seed=0, robust=True, n_byzantine=1),
            MLPAdapter(seed=0, robust=True, n_byzantine=1),
            GBDTAdapter()]
    e = StackedEnsemble(base, meta_rounds=3, robust=True, n_byzantine=1)
    e.fed_train(train_views, meta_frac=0.4, seed=6)
    p = e.predict_proba(make_test_views(n=500, seed=11))
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_degrades_gracefully_when_base_dropped():
    """Dropping a base model still yields a working ensemble whose recall is at
    least the best remaining single base (graceful degradation, no crash)."""
    train_views = make_bank_views(n_banks=4, n_per_bank=1000, seed=8)
    test_views = make_test_views(n=2500, seed=8888)
    y_test = test_views[0]["y"]

    # full roster
    full = fed_train_ensemble(train_views, ens.default_base_learners(seed=0),
                              meta_frac=0.4, seed=8)
    full_pred = full.predict_proba(test_views) > 0.5
    full_recall = float((full_pred & (y_test == 1)).sum() / (y_test == 1).sum())

    # drop the sequence (temporal) base -> still works, still >= remaining best
    reduced_learners = [LogisticAdapter(seed=0), MLPAdapter(seed=0),
                        GBDTAdapter(), EmbeddingAdapter(seed=0)]
    reduced = fed_train_ensemble(train_views, reduced_learners,
                                 meta_frac=0.4, seed=8)
    r_recs = _per_base_recall(reduced, test_views, y_test)
    r_pred = reduced.predict_proba(test_views) > 0.5
    r_recall = float((r_pred & (y_test == 1)).sum() / (y_test == 1).sum())

    # reduced ensemble is valid and at least matches its own best base
    assert reduced.predict_proba(test_views).min() >= 0.0
    assert r_recall >= max(r_recs.values()) - 1e-9
    # dropping a covering base should not improve recall over the full roster
    assert r_recall <= full_recall + 1e-9


def test_auc_helper_basic():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(auc(perfect, y) - 1.0) < 1e-9
    worst = np.array([0.9, 0.8, 0.2, 0.1])
    assert abs(auc(worst, y) - 0.0) < 1e-9
    assert abs(auc(np.array([0.5, 0.5, 0.5, 0.5]), y) - 0.5) < 1e-9
