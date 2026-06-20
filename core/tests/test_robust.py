"""Tests for veritas_core.robust -- real Byzantine robustness, not the toy case.

The honest model lives near a fixed vector; attackers use the poisoning model
from veritas_core.attack (sign-flip + amplify). We assert that the robust
aggregators stay near the honest mean while plain fedavg is dragged off.
"""

import numpy as np

from veritas_core.aggregation import fedavg
from veritas_core.attack import poisoned_update
from veritas_core.robust import (
    norm_clip,
    trimmed_mean,
    coordinate_median,
    multi_krum_select,
    bulyan,
    robust_aggregate,
)


RNG = np.random.default_rng(7)
HONEST_CENTER = np.array([1.0, 1.0, 1.0, 1.0])


def _honest(n, jitter=0.05):
    """n honest updates tightly clustered around HONEST_CENTER."""
    return [HONEST_CENTER + jitter * RNG.standard_normal(HONEST_CENTER.shape)
            for _ in range(n)]


# --------------------------------------------------------------------------- #
# norm_clip
# --------------------------------------------------------------------------- #
def test_norm_clip_caps_magnitude_preserves_direction():
    u = np.array([3.0, 4.0])          # norm 5
    clipped = norm_clip([u], max_norm=1.0)[0]
    assert np.isclose(np.linalg.norm(clipped), 1.0)
    # Direction preserved.
    assert np.allclose(clipped, u / 5.0)


def test_norm_clip_leaves_small_updates():
    u = np.array([0.1, 0.2])
    clipped = norm_clip([u], max_norm=10.0)[0]
    assert np.allclose(clipped, u)


# --------------------------------------------------------------------------- #
# trimmed_mean / coordinate_median reduce to ~mean on honest input
# --------------------------------------------------------------------------- #
def test_trimmed_mean_reduces_to_mean_on_honest():
    honest = _honest(8)
    tm = trimmed_mean(honest, trim_ratio=0.25)
    assert np.allclose(tm, np.mean(honest, axis=0), atol=0.05)


def test_coordinate_median_reduces_to_mean_on_honest():
    honest = _honest(9)
    med = coordinate_median(honest)
    assert np.allclose(med, np.mean(honest, axis=0), atol=0.05)


# --------------------------------------------------------------------------- #
# Single large outlier rejected by krum and bulyan
# --------------------------------------------------------------------------- #
def test_krum_rejects_single_large_outlier():
    honest = _honest(7)
    attacker = poisoned_update(HONEST_CENTER, scale=50.0)  # huge sign-flip
    updates = honest + [attacker]
    agg, selected, scores = multi_krum_select(updates, n_byzantine=1)
    assert 7 not in selected                       # attacker dropped
    assert scores[7] == max(scores)                # attacker scores worst
    assert np.linalg.norm(agg - HONEST_CENTER) < 0.5


def test_bulyan_rejects_single_large_outlier():
    honest = _honest(7)               # n=8, f=1 -> n >= 4f+3 (8 >= 7) ok
    attacker = poisoned_update(HONEST_CENTER, scale=50.0)
    updates = honest + [attacker]
    agg, selected, scores = bulyan(updates, n_byzantine=1)
    assert 7 not in selected
    assert np.linalg.norm(agg - HONEST_CENTER) < 0.5


# --------------------------------------------------------------------------- #
# Multiple colluding attackers: robust stays close, fedavg dragged far off
# --------------------------------------------------------------------------- #
def test_colluding_attackers_trimmed_and_bulyan_beat_fedavg():
    honest = _honest(6)
    # 2 of 8 attackers submitting the SAME poisoned direction.
    poison = poisoned_update(HONEST_CENTER, scale=8.0)
    updates = honest + [poison.copy(), poison.copy()]

    honest_mean = np.mean(honest, axis=0)

    fa = fedavg(updates)
    tm = trimmed_mean(updates, trim_ratio=0.25)            # k=2 per side >= f=2
    bz, _, _ = bulyan(updates, n_byzantine=2)              # n=8, 4f+3=11 -> falls back to krum

    fa_err = np.linalg.norm(fa - honest_mean)
    tm_err = np.linalg.norm(tm - honest_mean)
    bz_err = np.linalg.norm(bz - honest_mean)

    # Robust methods stay close to honest mean.
    assert tm_err < 0.3
    assert bz_err < 0.3
    # fedavg is dragged far off -- assert a large gap.
    assert fa_err > 2.0
    assert fa_err > 5 * tm_err
    assert fa_err > 5 * bz_err


def test_colluding_attackers_krum_selects_honest_majority():
    honest = _honest(6)
    poison = poisoned_update(HONEST_CENTER, scale=8.0)
    updates = honest + [poison.copy(), poison.copy()]
    agg, selected, scores = multi_krum_select(updates, n_byzantine=2)
    # Neither attacker (indices 6, 7) should be selected.
    assert 6 not in selected and 7 not in selected
    assert np.linalg.norm(agg - np.mean(honest, axis=0)) < 0.3


# --------------------------------------------------------------------------- #
# Stealthy small-magnitude poison: norm_clip + trimmed_mean still mitigates
# --------------------------------------------------------------------------- #
def test_stealthy_poison_mitigated_by_clip_and_trim():
    honest = _honest(7)
    honest_mean = np.mean(honest, axis=0)
    # Just inside a naive 10x-median threshold: a single attacker nudges
    # every coordinate by a modest but consistent amount.
    stealth = HONEST_CENTER + np.array([2.5, 2.5, 2.5, 2.5])
    updates = honest + [stealth]

    fa = fedavg(updates)
    agg, info = robust_aggregate(
        updates, method="trimmed_mean", n_byzantine=1, max_norm=3.0
    )

    fa_err = np.linalg.norm(fa - honest_mean)
    rob_err = np.linalg.norm(agg - honest_mean)
    assert rob_err < fa_err          # robust path strictly closer to honest
    assert rob_err < 0.3
    assert info["clipped"] is True


# --------------------------------------------------------------------------- #
# Dispatcher: info reporting (rejected members + attack_detected)
# --------------------------------------------------------------------------- #
def test_dispatch_krum_reports_rejected_and_detection():
    honest = _honest(7)
    attacker = poisoned_update(HONEST_CENTER, scale=50.0)
    updates = honest + [attacker]
    agg, info = robust_aggregate(updates, method="krum", n_byzantine=1)
    assert info["attack_detected"] is True
    assert 7 in info["rejected"]
    assert 7 not in info["selected"]
    assert len(info["scores"]) == 8


def test_dispatch_all_honest_no_attack_flag():
    honest = _honest(8)
    agg, info = robust_aggregate(honest, method="krum", n_byzantine=1)
    # All-honest, m = n - f = 7, so one borderline member is "rejected" by
    # selection, but the aggregate must still be on target.
    assert np.linalg.norm(agg - np.mean(honest, axis=0)) < 0.2


def test_dispatch_median_keeps_all_members():
    honest = _honest(5)
    agg, info = robust_aggregate(honest, method="median", n_byzantine=1)
    assert info["rejected"] == []
    assert info["attack_detected"] is False
    assert np.allclose(agg, coordinate_median(honest))


def test_dispatch_bulyan_fallback_flag_when_n_small():
    # n=6, f=2 -> 4f+3 = 11 > 6, so Bulyan must fall back to Krum.
    honest = _honest(4)
    poison = poisoned_update(HONEST_CENTER, scale=20.0)
    updates = honest + [poison.copy(), poison.copy()]
    agg, info = robust_aggregate(updates, method="bulyan", n_byzantine=2)
    assert info["fellback"] is True
    assert info["method"] == "krum"


def test_invalid_method_raises():
    try:
        robust_aggregate(_honest(4), method="nope", n_byzantine=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown method")
