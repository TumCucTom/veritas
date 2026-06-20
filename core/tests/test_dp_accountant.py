"""Tests for the RDP privacy accountant (veritas_core.dp_accountant)."""

import math

import pytest

from veritas_core.dp_accountant import (
    RDPAccountant,
    calibrate_noise_multiplier,
    dp_summary,
    gaussian_rdp,
    rdp_to_epsilon,
    subsampled_gaussian_rdp,
)

DELTA = 1e-5


def _eps(noise_multiplier, num_rounds=1, sample_rate=1.0, delta=DELTA):
    acct = RDPAccountant()
    for _ in range(num_rounds):
        acct.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return acct.get_epsilon(delta)


def test_gaussian_rdp_formula():
    # rdp(alpha) = alpha / (2 sigma^2)
    assert gaussian_rdp(2.0, 1.0) == pytest.approx(2.0 / (2.0 * 1.0))
    assert gaussian_rdp(10.0, 2.0) == pytest.approx(10.0 / (2.0 * 4.0))
    # sigma <= 0 => no privacy
    assert gaussian_rdp(2.0, 0.0) == math.inf


def test_sigma_one_sanity_point():
    # sigma=1.0, delta=1e-5, 1 round => single-digit epsilon.
    eps = _eps(1.0, num_rounds=1)
    assert 1.0 < eps < 10.0, f"expected single-digit eps, got {eps}"


def test_monotonic_in_rounds():
    e1 = _eps(1.0, num_rounds=1)
    e10 = _eps(1.0, num_rounds=10)
    e100 = _eps(1.0, num_rounds=100)
    assert e1 < e10 < e100


def test_monotonic_in_sigma():
    # smaller sigma => more leakage => larger epsilon
    e_small = _eps(0.5, num_rounds=10)
    e_mid = _eps(1.0, num_rounds=10)
    e_big = _eps(4.0, num_rounds=10)
    assert e_small > e_mid > e_big


def test_tiny_sigma_blows_up_huge_sigma_vanishes():
    eps_tiny = _eps(1e-3, num_rounds=5)
    eps_huge = _eps(1e6, num_rounds=5)
    assert eps_tiny > 1e3
    # Huge sigma drives epsilon toward zero. With a finite grid of RDP orders
    # the conversion floors at a small positive residual (~log(1/delta)/(a_max-1)),
    # so assert "small", not literally zero.
    assert eps_huge < 0.05
    assert eps_huge < eps_tiny


def test_calibrate_account_round_trip():
    target = 3.0
    rounds = 50
    sigma = calibrate_noise_multiplier(target, DELTA, rounds)
    assert sigma > 0
    eps = _eps(sigma, num_rounds=rounds)
    # accounting the calibrated sigma must not exceed the target budget
    assert eps <= target + 1e-3
    # and it should be reasonably tight (not wildly over-noised)
    assert eps >= target - 0.5


def test_calibrate_round_trip_subsampled():
    target = 2.0
    rounds = 100
    q = 0.1
    sigma = calibrate_noise_multiplier(target, DELTA, rounds, sample_rate=q)
    acct = RDPAccountant()
    for _ in range(rounds):
        acct.step(noise_multiplier=sigma, sample_rate=q)
    eps = acct.get_epsilon(DELTA)
    assert eps <= target + 1e-3


def test_calibrate_monotonic_in_target():
    # tighter budget (smaller target eps) requires more noise (larger sigma)
    s_loose = calibrate_noise_multiplier(8.0, DELTA, 20)
    s_tight = calibrate_noise_multiplier(1.0, DELTA, 20)
    assert s_tight > s_loose


def test_subsampling_amplifies_privacy():
    # With the same sigma and rounds, subsampling (q<1) should give smaller eps
    # than full-batch (q=1).
    full = _eps(1.0, num_rounds=50, sample_rate=1.0)
    sub = _eps(1.0, num_rounds=50, sample_rate=0.05)
    assert sub < full


def test_q_one_matches_nonsubsampled():
    for alpha in (2.0, 5.0, 17.0):
        a = gaussian_rdp(alpha, 1.3)
        b = subsampled_gaussian_rdp(alpha, 1.3, 1.0)
        assert a == pytest.approx(b)


def test_rdp_to_epsilon_picks_min_order():
    sigma = 1.0
    rdp = [(alpha, gaussian_rdp(alpha, sigma)) for alpha in (1.5, 2.0, 4.0, 8.0, 32.0)]
    eps, alpha = rdp_to_epsilon(rdp, DELTA)
    assert eps > 0
    assert alpha in {1.5, 2.0, 4.0, 8.0, 32.0}


def test_dp_summary_shape():
    s = dp_summary(1.0, DELTA, 10)
    assert set(s) == {"epsilon", "delta", "noise_multiplier", "rounds", "sample_rate"}
    assert s["rounds"] == 10
    assert s["noise_multiplier"] == 1.0
    assert s["delta"] == DELTA
    assert s["epsilon"] == pytest.approx(_eps(1.0, num_rounds=10))


def test_calibrate_infeasible_raises():
    with pytest.raises(RuntimeError):
        # impossibly tight budget for many rounds, capped sigma_max
        calibrate_noise_multiplier(1e-6, DELTA, 10000, sigma_max=10.0)


def test_bad_args():
    with pytest.raises(ValueError):
        calibrate_noise_multiplier(-1.0, DELTA, 10)
    with pytest.raises(ValueError):
        calibrate_noise_multiplier(1.0, 0.0, 10)
    with pytest.raises(ValueError):
        calibrate_noise_multiplier(1.0, DELTA, 0)
    with pytest.raises(ValueError):
        calibrate_noise_multiplier(1.0, DELTA, 10, sample_rate=2.0)
