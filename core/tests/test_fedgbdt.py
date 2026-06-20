import numpy as np

from veritas_core.fedgbdt import (
    fed_train_gbdt, predict_proba, recall, secure_sum, secure_mask,
    establish_secret_seeds, _pair_mask,
)
from veritas_core import model as logmodel


def _xor_interaction(n, seed):
    """Non-linear XOR-style interaction pattern that a LINEAR logistic model
    cannot separate but a tree ensemble can. Label = XOR(f0>0, f1>0) with noise;
    remaining features are noise. No single linear boundary works."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 8))
    pos = (X[:, 0] > 0) ^ (X[:, 1] > 0)
    flip = rng.random(n) < 0.05
    y = (pos ^ flip).astype(np.int64)
    return X.astype(np.float64), y


def _split_banks(X, y, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    return [(X[s], y[s]) for s in np.array_split(idx, k)]


def test_predict_proba_in_range():
    X, y = _xor_interaction(2000, seed=1)
    banks = _split_banks(X, y, 4, seed=1)
    m = fed_train_gbdt(banks, n_trees=15, max_depth=3, n_bins=16, lr=0.3)
    p = predict_proba(m, X)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_gbdt_beats_logistic_on_interaction():
    X, y = _xor_interaction(4000, seed=2)
    banks = _split_banks(X, y, 4, seed=2)
    m = fed_train_gbdt(banks, n_trees=30, max_depth=3, n_bins=16, lr=0.3)
    gbdt_rec = recall(m, X, y)

    w = logmodel.train_local(logmodel.init_weights(X.shape[1]), X, y,
                             epochs=200, lr=0.3)
    log_rec = logmodel.recall(w, X, y)
    assert gbdt_rec > log_rec + 0.2, (gbdt_rec, log_rec)


def test_fed_matches_centralised():
    X, y = _xor_interaction(4000, seed=3)
    banks = _split_banks(X, y, 5, seed=3)
    fed = fed_train_gbdt(banks, n_trees=25, max_depth=3, n_bins=16, lr=0.3)
    central = fed_train_gbdt([(X, y)], n_trees=25, max_depth=3, n_bins=16, lr=0.3)
    fed_rec = recall(fed, X, y)
    cen_rec = recall(central, X, y)
    assert fed_rec >= cen_rec - 0.05, (fed_rec, cen_rec)


def test_secure_mask_hides_individual_but_sums_true():
    rng = np.random.default_rng(0)
    raw = [rng.normal(0, 1, (3, 16)) for _ in range(4)]
    masked = secure_mask(raw, session=42)
    # individual masked histogram differs from the raw local histogram
    for r, m in zip(raw, masked):
        assert not np.allclose(r, m)
    # but the secure sum equals the true total
    true_total = sum(raw)
    agg = secure_sum(raw, secure=True, session=42)
    assert np.allclose(agg, true_total, atol=1e-6)
    # and the masked stack still sums to the true total
    assert np.allclose(sum(masked), true_total, atol=1e-6)


def test_masks_keyed_off_secret_not_public_indices():
    """An attacker who knows ONLY the public indices/session/shape cannot
    recompute a pairwise mask: the mask is keyed off a per-pair SECRET seed.

    This is the security property the old (public-index-seeded) implementation
    violated — anyone could regenerate _pair_mask(i, j, session) and unmask a
    single bank's histogram.
    """
    shape = (3, 16)
    session = 42
    seeds = establish_secret_seeds(2)
    real = _pair_mask(seeds[(0, 1)], shape, session)

    # Attacker tries to forge the mask from public info only (guessing the
    # secret). A different secret yields a totally different mask.
    forged = _pair_mask(b"\x00" * 32, shape, session)
    assert not np.allclose(real, forged)

    # Two independently-dealt secret tables disagree -> the mask is not a
    # deterministic function of the public (indices, session, shape).
    other = establish_secret_seeds(2)
    real2 = _pair_mask(other[(0, 1)], shape, session)
    assert not np.allclose(real, real2)


def test_single_party_cannot_unmask_another():
    """Masks cancel in aggregate, but a single party (or the coordinator) cannot
    peel another party's mask without that pair's secret seed."""
    rng = np.random.default_rng(3)
    raw = [rng.normal(0, 1, (3, 16)) for _ in range(3)]
    seeds = establish_secret_seeds(3, master_rng=np.random.default_rng(9))
    masked = secure_mask(raw, session=7, seeds=seeds)

    # Aggregate cancels to the true total.
    assert np.allclose(sum(masked), sum(raw), atol=1e-6)

    # Bank 2 holds only the seeds for pairs that include itself: (0,2) and (1,2).
    # It does NOT know (0,1), so it cannot reconstruct the mask hiding bank 0
    # from bank 1, and thus cannot recover bank 0's or bank 1's raw histogram.
    assert (0, 1) in seeds  # exists in the dealer table...
    # ...but a party lacking it cannot regenerate that mask:
    guess = _pair_mask(b"\x11" * 32, raw[0].shape, 7)
    # subtracting a wrong guess does not reveal bank 0's raw tensor
    assert not np.allclose(masked[0] - guess, raw[0], atol=1e-3)


def test_secure_training_matches_insecure():
    X, y = _xor_interaction(2000, seed=5)
    banks = _split_banks(X, y, 4, seed=5)
    sec = fed_train_gbdt(banks, n_trees=15, max_depth=3, secure=True)
    ins = fed_train_gbdt(banks, n_trees=15, max_depth=3, secure=False)
    # same tree structure / leaves up to float tolerance -> same recall
    assert abs(recall(sec, X, y) - recall(ins, X, y)) < 1e-9
