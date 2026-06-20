"""Differential-privacy primitives: clip + Gaussian noise on an update.

SPEC-DP (randomness policy)
---------------------------
Production noise MUST come from a cryptographically secure RNG. The default
noise generator here is seeded from ``secrets.randbits(128)`` (os.urandom under
the hood), so no constant seed ever drives privacy noise on the default path.

Deterministic seeding is allowed ONLY when the caller passes an explicit ``rng``
(test-only): tests pass ``np.random.default_rng(<seed>)`` to get reproducible
noise. The default code path never fabricates a fixed seed.

The privacy-noise RNG is kept SEPARATE from any synthetic-data RNG used to
fabricate demo/training data — callers should not reuse one generator for both.
"""
import secrets

import numpy as np


def default_noise_rng() -> np.random.Generator:
    """A fresh CSPRNG-seeded numpy Generator for privacy NOISE.

    Seeded from ``secrets.randbits(128)`` (cryptographically secure entropy), so
    the default noise path is non-deterministic and never keyed off a constant.
    Use a dedicated generator for noise; do not share it with data synthesis.
    """
    return np.random.default_rng(secrets.randbits(128))


def clip_update(u, max_norm):
    nrm = float(np.linalg.norm(u))
    return u if nrm <= max_norm else u * (max_norm / nrm)


def add_noise(u, sigma, max_norm, rng=None):
    """Add N(0, (sigma*max_norm)^2) noise per coordinate.

    ``rng`` defaults to a CSPRNG-seeded generator (SPEC-DP). Pass an explicit
    seeded ``np.random.default_rng(seed)`` only for deterministic tests.
    """
    if rng is None:
        rng = default_noise_rng()
    return u + rng.normal(0, sigma * max_norm, size=u.shape)


def privatize(u, max_norm, sigma, rng=None):
    """clip then add Gaussian noise; ``rng`` defaults to a secure generator."""
    return add_noise(clip_update(u, max_norm), sigma, max_norm, rng)
