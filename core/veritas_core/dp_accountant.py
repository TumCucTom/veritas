"""Renyi Differential Privacy (RDP) accountant for the Gaussian mechanism.

This module gives the clip+noise mechanism in ``veritas_core.dp`` a rigorous
privacy story. The existing mechanism adds Gaussian noise with stddev
``sigma * max_norm`` to an L2-clipped update, so ``sigma`` is exactly the
*noise multiplier* (noise stddev divided by the clipping / L2-sensitivity
bound) that the Gaussian-mechanism RDP theory is parameterised by.

We track privacy in RDP space and convert to ``(epsilon, delta)``-DP at the
end. This is the standard modern accounting approach (Mironov 2017; Mironov,
Talwar, Zhang 2019 for the subsampled case) and is what Opacus / TF-Privacy
use under the hood.

Key formulas
------------
Non-subsampled Gaussian mechanism, RDP at order ``alpha > 1``::

    rdp(alpha) = alpha / (2 * noise_multiplier ** 2)

Composition of ``k`` mechanisms simply adds RDP at each order::

    rdp_total(alpha) = sum_i rdp_i(alpha)

Subsampled (Poisson, rate ``q``) Gaussian mechanism RDP is computed with the
numerically-stable log-domain binomial expansion of Mironov, Talwar & Zhang
(2019), the same series used by TF-Privacy / Opacus.

RDP -> (epsilon, delta) conversion. For each order ``alpha`` we use the tight
bound of Mironov (2017, Prop. 3) / Balle et al. (2020)::

    eps(alpha) = rdp(alpha)
                 + log((alpha - 1) / alpha)
                 - (log(delta) + log(alpha)) / (alpha - 1)

and minimise over the grid of orders. The simpler classical bound
``eps = rdp(alpha) + log(1/delta)/(alpha-1)`` is always an upper bound on the
above and would also be sound; we use the tighter form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "DEFAULT_ORDERS",
    "gaussian_rdp",
    "subsampled_gaussian_rdp",
    "rdp_to_epsilon",
    "RDPAccountant",
    "calibrate_noise_multiplier",
    "dp_summary",
]


def _default_orders() -> List[float]:
    """A broad grid of RDP orders alpha > 1.

    Fine grid at small orders (which dominate for large noise / small eps)
    plus a coarse tail of large integer orders (which dominate for small
    noise / subsampling amplification).
    """
    orders = [1.0 + x / 10.0 for x in range(1, 100)]  # 1.1 .. 10.9
    orders += list(range(11, 64))  # 11 .. 63
    orders += [128.0, 256.0, 512.0]
    return orders


DEFAULT_ORDERS: List[float] = _default_orders()


def gaussian_rdp(alpha: float, noise_multiplier: float) -> float:
    """RDP at order ``alpha`` of the (non-subsampled) Gaussian mechanism.

    The Gaussian mechanism with sensitivity 1 and noise stddev
    ``noise_multiplier`` satisfies ``(alpha, alpha / (2 * sigma^2))``-RDP for
    every order ``alpha > 1``.

    Returns ``+inf`` for a non-positive noise multiplier (no privacy).
    """
    if noise_multiplier <= 0.0:
        return math.inf
    if alpha <= 1.0:
        raise ValueError("RDP order alpha must be > 1")
    return alpha / (2.0 * noise_multiplier * noise_multiplier)


def _log_add(log_a: float, log_b: float) -> float:
    """Stable ``log(exp(log_a) + exp(log_b))``."""
    if log_a == -math.inf:
        return log_b
    if log_b == -math.inf:
        return log_a
    if log_a < log_b:
        log_a, log_b = log_b, log_a
    return log_a + math.log1p(math.exp(log_b - log_a))


def _log_sub(log_a: float, log_b: float) -> float:
    """Stable ``log(exp(log_a) - exp(log_b))`` (requires log_a >= log_b)."""
    if log_b == -math.inf:
        return log_a
    if log_a <= log_b:
        return -math.inf
    return log_a + math.log1p(-math.exp(log_b - log_a))


def _log_comb(n: int, k: int) -> float:
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
    )


def _subsampled_rdp_integer_order(
    alpha_int: int, q: float, sigma: float
) -> float:
    """RDP of the Poisson-subsampled Gaussian mechanism at integer order.

    Implements the log-domain binomial expansion of Mironov, Talwar & Zhang
    (2019), "Renyi Differential Privacy of the Sampled Gaussian Mechanism".
    """
    # log of each term in the moment expansion, then logsumexp.
    log_q = math.log(q)
    log_1mq = math.log1p(-q)
    log_terms: List[float] = []
    for i in range(alpha_int + 1):
        log_coef = _log_comb(alpha_int, i) + i * log_q + (alpha_int - i) * log_1mq
        # E[ exp( (i^2 - i) / (2 sigma^2) ) ] contribution
        exponent = (i * i - i) / (2.0 * sigma * sigma)
        log_terms.append(log_coef + exponent)
    log_moment = log_terms[0]
    for t in log_terms[1:]:
        log_moment = _log_add(log_moment, t)
    return log_moment / (alpha_int - 1)


def subsampled_gaussian_rdp(
    alpha: float, noise_multiplier: float, sample_rate: float
) -> float:
    """RDP at order ``alpha`` of the Poisson-subsampled Gaussian mechanism.

    ``sample_rate`` (``q``) is the per-round Poisson sampling probability. For
    ``q == 1`` this reduces exactly to :func:`gaussian_rdp`.

    For non-integer ``alpha`` we evaluate the expansion at the two bracketing
    integer orders and take the larger (a valid upper bound), which keeps the
    accountant sound on the default fractional grid.
    """
    if noise_multiplier <= 0.0:
        return math.inf
    if alpha <= 1.0:
        raise ValueError("RDP order alpha must be > 1")
    q = sample_rate
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return gaussian_rdp(alpha, noise_multiplier)

    lo = int(math.floor(alpha))
    if lo < 2:
        lo = 2
    hi = lo + 1
    rdp_lo = _subsampled_rdp_integer_order(lo, q, noise_multiplier)
    rdp_hi = _subsampled_rdp_integer_order(hi, q, noise_multiplier)
    return max(rdp_lo, rdp_hi)


def _rdp_at_order(
    alpha: float, noise_multiplier: float, sample_rate: float
) -> float:
    if sample_rate >= 1.0:
        return gaussian_rdp(alpha, noise_multiplier)
    return subsampled_gaussian_rdp(alpha, noise_multiplier, sample_rate)


def _eps_from_rdp_at_order(rdp: float, alpha: float, delta: float) -> float:
    """Tight RDP -> eps conversion at a single order (Mironov 2017 / Balle 2020)."""
    if rdp == math.inf:
        return math.inf
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    # eps = rdp + log((alpha-1)/alpha) - (log(delta) + log(alpha)) / (alpha-1)
    eps = (
        rdp
        + math.log((alpha - 1.0) / alpha)
        - (math.log(delta) + math.log(alpha)) / (alpha - 1.0)
    )
    return max(eps, 0.0)


def rdp_to_epsilon(
    rdp_by_order: Sequence[Tuple[float, float]], delta: float
) -> Tuple[float, float]:
    """Convert accumulated RDP to ``(epsilon, best_alpha)``.

    ``rdp_by_order`` is a sequence of ``(alpha, rdp(alpha))`` pairs. We return
    the minimum ``eps`` over orders and the order that achieved it.
    """
    best_eps = math.inf
    best_alpha = math.nan
    for alpha, rdp in rdp_by_order:
        eps = _eps_from_rdp_at_order(rdp, alpha, delta)
        if eps < best_eps:
            best_eps = eps
            best_alpha = alpha
    return best_eps, best_alpha


@dataclass
class RDPAccountant:
    """Accumulates Renyi DP across federation rounds.

    Usage::

        acct = RDPAccountant()
        for _ in range(num_rounds):
            acct.step(noise_multiplier=sigma, sample_rate=q)
        spent_eps = acct.get_epsilon(delta=1e-5)
    """

    orders: List[float] = field(default_factory=lambda: list(DEFAULT_ORDERS))
    _rdp: np.ndarray = field(init=False, repr=False)
    steps: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.orders:
            raise ValueError("orders must be non-empty")
        if any(a <= 1.0 for a in self.orders):
            raise ValueError("all RDP orders must be > 1")
        self._rdp = np.zeros(len(self.orders), dtype=float)

    def step(self, noise_multiplier: float, sample_rate: float = 1.0) -> "RDPAccountant":
        """Account one round of the Gaussian mechanism. Returns self (chainable)."""
        for i, alpha in enumerate(self.orders):
            self._rdp[i] += _rdp_at_order(alpha, noise_multiplier, sample_rate)
        self.steps += 1
        return self

    def steps_taken(self) -> int:
        return self.steps

    def rdp_by_order(self) -> List[Tuple[float, float]]:
        return list(zip(self.orders, self._rdp.tolist()))

    def get_epsilon(self, delta: float) -> float:
        """Convert the accumulated RDP to ``(epsilon, delta)``-DP."""
        eps, _ = rdp_to_epsilon(self.rdp_by_order(), delta)
        return eps

    def get_epsilon_and_order(self, delta: float) -> Tuple[float, float]:
        return rdp_to_epsilon(self.rdp_by_order(), delta)


def _epsilon_for_sigma(
    noise_multiplier: float,
    delta: float,
    num_rounds: int,
    sample_rate: float,
    orders: Sequence[float],
) -> float:
    rdp = [
        (alpha, num_rounds * _rdp_at_order(alpha, noise_multiplier, sample_rate))
        for alpha in orders
    ]
    eps, _ = rdp_to_epsilon(rdp, delta)
    return eps


def calibrate_noise_multiplier(
    target_epsilon: float,
    delta: float,
    num_rounds: int,
    sample_rate: float = 1.0,
    orders: Sequence[float] | None = None,
    sigma_max: float = 1024.0,
    tol: float = 1e-4,
) -> float:
    """Find the smallest noise multiplier sigma meeting ``(target_epsilon, delta)``.

    Composing ``num_rounds`` Gaussian-mechanism rounds (optionally Poisson
    subsampled at ``sample_rate``) must satisfy ``(target_epsilon, delta)``-DP.
    epsilon is monotonically decreasing in sigma, so we binary-search sigma.

    Returns the calibrated noise multiplier. Raises ``ValueError`` on bad
    arguments; raises ``RuntimeError`` if even ``sigma_max`` cannot reach the
    target (target_epsilon too small for the requested rounds).
    """
    if target_epsilon <= 0.0:
        raise ValueError("target_epsilon must be > 0")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if num_rounds < 1:
        raise ValueError("num_rounds must be >= 1")
    if not 0.0 < sample_rate <= 1.0:
        raise ValueError("sample_rate must be in (0, 1]")
    grid = list(orders) if orders is not None else list(DEFAULT_ORDERS)

    def eps_of(sigma: float) -> float:
        return _epsilon_for_sigma(sigma, delta, num_rounds, sample_rate, grid)

    lo, hi = 1e-6, sigma_max
    if eps_of(hi) > target_epsilon:
        raise RuntimeError(
            f"sigma_max={sigma_max} insufficient for target_epsilon={target_epsilon} "
            f"over {num_rounds} rounds (sample_rate={sample_rate}); "
            f"raise sigma_max or relax the budget."
        )
    # Invariant: eps(lo) >= target > eps(hi). Shrink the bracket.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if eps_of(mid) > target_epsilon:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * hi:
            break
    # Return hi: it satisfies eps(hi) <= target_epsilon (sound side).
    return hi


def dp_summary(
    noise_multiplier: float,
    delta: float,
    num_rounds: int,
    sample_rate: float = 1.0,
    orders: Sequence[float] | None = None,
) -> Dict[str, float]:
    """Report the ``(epsilon, delta)`` spent by a given configuration.

    Convenience wrapper for dashboards / control-plane reporting.
    """
    grid = list(orders) if orders is not None else list(DEFAULT_ORDERS)
    eps = _epsilon_for_sigma(noise_multiplier, delta, num_rounds, sample_rate, grid)
    return {
        "epsilon": eps,
        "delta": delta,
        "noise_multiplier": float(noise_multiplier),
        "rounds": int(num_rounds),
        "sample_rate": float(sample_rate),
    }
