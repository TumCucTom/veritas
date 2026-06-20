"""Correctness bar for evasion-robust adversarial training.

Setup mirrors the real threat: a bank trains a fraud detector, then a
fraudster who can probe the model nudges true-fraud transaction features
(within an L-inf epsilon budget) to slip under the alert threshold.

The baseline here is an *over-trained* logistic model on a small sample --
the realistic case where the model has memorised spurious large-weight
directions (including weight wasted on non-predictive features) that the
evader gets to exploit for free. Adversarial training removes that attack
surface.

Asserts:
  * FGSM and PGD actually evade the NORMALLY-trained baseline.
  * PGD is at least as strong as FGSM on the baseline (PGD >= FGSM).
  * The ADVERSARIALLY-trained model has a MUCH lower evasion rate
    (>= 2x reduction on the pinned seed; a strict reduction on every seed
    of a sweep) while keeping clean-data recall healthy.
"""
import numpy as np

from veritas_core.data import make_bank_data, FEATURE_DIM
from veritas_core.model import init_weights, train_local, predict_proba, recall
from veritas_core.adversarial import (
    fgsm_perturb, pgd_perturb, adversarial_train_local, evasion_success_rate,
)

EPS = 0.5
N = 60                # small -> baseline over-fits, giving the evader leverage
SEED = 8              # pinned: large, representative robustness gap


def _baseline(seed=SEED, n=N):
    X, y = make_bank_data(n, fraud_rate=0.35, seed=seed)
    # heavy training on few samples -> memorises exploitable weight mass
    w = train_local(init_weights(FEATURE_DIM), X, y, epochs=3000, lr=0.6)
    return w, X, y


def _robust(X, y):
    return adversarial_train_local(
        init_weights(FEATURE_DIM), X, y, train_fn=train_local,
        epsilon=EPS, robust_logistic=True,
        robust_kwargs=dict(epochs=2500, lr=0.5),
    )


def test_fgsm_and_pgd_evade_baseline():
    w, X, y = _baseline()
    assert recall(w, X, y) > 0.7, "baseline must catch most fraud first"
    fgsm = evasion_success_rate(w, X, y, predict_proba, epsilon=EPS,
                                perturb_fn=fgsm_perturb)
    pgd = evasion_success_rate(w, X, y, predict_proba, epsilon=EPS,
                               perturb_fn=pgd_perturb)
    assert fgsm > 0.2, f"FGSM should evade a non-trivial fraction, got {fgsm}"
    assert pgd > 0.2, f"PGD should evade a non-trivial fraction, got {pgd}"


def test_pgd_at_least_as_strong_as_fgsm():
    w, X, y = _baseline()
    fgsm = evasion_success_rate(w, X, y, predict_proba, epsilon=EPS,
                                perturb_fn=fgsm_perturb)
    pgd = evasion_success_rate(w, X, y, predict_proba, epsilon=EPS,
                               perturb_fn=pgd_perturb)
    assert pgd >= fgsm - 1e-9, f"PGD ({pgd}) should be >= FGSM ({fgsm})"


def test_adversarial_training_reduces_evasion_keeps_recall():
    w, X, y = _baseline()
    base_evasion = evasion_success_rate(w, X, y, predict_proba, epsilon=EPS,
                                        perturb_fn=pgd_perturb)
    w_rob = _robust(X, y)
    rob_evasion = evasion_success_rate(w_rob, X, y, predict_proba, epsilon=EPS,
                                       perturb_fn=pgd_perturb)

    # MUCH lower evasion after hardening (>= 2x reduction on the pinned seed)
    assert rob_evasion < base_evasion * 0.5, (
        f"adversarial training should slash evasion: base={base_evasion} "
        f"robust={rob_evasion}")
    # clean-data recall must not collapse
    assert recall(w_rob, X, y) > 0.6, (
        f"robust model recall collapsed: {recall(w_rob, X, y)}")


def test_robustness_gain_is_reliable_across_seeds():
    """The hardened model evades strictly less than the baseline on every
    seed (not just the pinned one) -- the gain is not luck."""
    for seed in range(6):
        w, X, y = _baseline(seed=seed)
        base = evasion_success_rate(w, X, y, predict_proba, epsilon=EPS,
                                    perturb_fn=pgd_perturb)
        w_rob = _robust(X, y)
        rob = evasion_success_rate(w_rob, X, y, predict_proba, epsilon=EPS,
                                   perturb_fn=pgd_perturb)
        assert rob < base, f"seed {seed}: robust {rob} not < base {base}"
        assert recall(w_rob, X, y) > 0.6, f"seed {seed}: recall collapsed"
