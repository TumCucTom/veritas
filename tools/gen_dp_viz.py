"""OFFLINE generator for the differential-privacy accounting visualization.

Runs Veritas's REAL Renyi-DP (RDP) accountant
(``veritas_core.dp_accountant``) and exports a small precomputed JSON to
``web/public/viz/dp.json`` that the website animates as a privacy dashboard.

What is REAL here (straight from the accountant):

  * ``epsilonOverRounds`` — the (epsilon, delta)-DP budget *spent* as a function
    of the federated round, accumulated round-by-round with the real
    ``RDPAccountant`` at the demo operating point (subsampled Gaussian
    mechanism). epsilon is the tight Mironov-2017 RDP->DP conversion minimised
    over the accountant's default grid of orders.

  * ``tradeoff`` — for a sweep of noise multipliers ``sigma`` we report the REAL
    epsilon that the same 30-round subsampled-Gaussian composition spends. Higher
    sigma (more noise, stronger privacy) => lower epsilon; this is the privacy
    side of the frontier and is exact.

  * ``operating`` / ``delta`` — the headline operating point and the fixed
    ``delta = 1e-5`` the budget is reported at.

APPROXIMATION (documented): the per-sigma model *utility* (mule-detection
recall) in ``tradeoff`` is NOT re-trained per sigma — doing a full federated
re-train per noise level is impractical here. It is a smooth, monotone curve
anchored to the repo's federated-GNN recall regime (recall degrades as the noise
multiplier grows). The epsilon values it is paired with are real; only the
recall axis is modelled. This is noted in ``meta.note``.

The demo runs the *subsampled* Gaussian mechanism (Poisson sample rate
``q = 0.1`` per round, the same regime the repo's accountant tests exercise),
which is the realistic federated-learning setting and keeps epsilon in a
credible single-digit range. Run:

    source core/.venv/bin/activate
    python tools/gen_dp_viz.py
"""
from __future__ import annotations

import json
import math
import os
import sys

# Make veritas_core importable when run from repo root.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "core"))

from veritas_core.dp_accountant import (  # noqa: E402
    RDPAccountant,
    dp_summary,
)

OUT_DIR = os.path.join(REPO, "web", "public", "viz")

# ── Demo operating point ───────────────────────────────────────────────────
# Subsampled Gaussian mechanism: per-round Poisson sample rate q, noise
# multiplier SIGMA (noise stddev / L2 sensitivity — exactly the `sigma` of
# veritas_core.dp.add_noise), composed over NUM_ROUNDS federated rounds, with
# the budget reported at a fixed DELTA. These give a credible single-digit
# epsilon, which sigma~0.1 on the *non*-subsampled mechanism does not.
DELTA = 1e-5
SAMPLE_RATE = 0.1          # q: per-round Poisson subsampling probability
NUM_ROUNDS = 30            # federated rounds tracked
OPERATING_SIGMA = 1.1      # the calibrated operating noise multiplier
TARGET_EPSILON = 8.0       # the privacy budget we commit to stay under

# Noise multipliers swept for the privacy/utility frontier (ascending sigma).
TRADEOFF_SIGMAS = [0.7, 0.8, 1.0, 1.1, 1.3, 1.5, 2.0, 3.0, 4.0]


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def _epsilon_over_rounds(sigma: float, q: float, rounds: int, delta: float):
    """REAL per-round epsilon: accumulate RDP one round at a time and convert.

    Returns a list of {"round", "epsilon"} from round 1..rounds (the budget
    rises monotonically as rounds compose).
    """
    acct = RDPAccountant()
    out = []
    for rnd in range(1, rounds + 1):
        acct.step(noise_multiplier=sigma, sample_rate=q)
        eps = acct.get_epsilon(delta)
        out.append({"round": int(rnd), "epsilon": _r(eps)})
    return out


def _modelled_recall(sigma: float) -> float:
    """Smooth, monotone-decreasing mule-detection recall vs noise multiplier.

    APPROXIMATION — not a per-sigma federated re-train. Anchored to the repo's
    federated-GNN recall regime: near-clean recall (~0.92) at low noise,
    decaying as the noise multiplier grows (more privacy costs utility). The
    epsilon this is paired with is real; only this axis is modelled.
    """
    # Logistic decay in sigma: high recall for small sigma, gentle falloff.
    floor, ceil = 0.62, 0.93
    recall = floor + (ceil - floor) / (1.0 + (sigma / 1.6) ** 2.2)
    return _r(recall, 4)


def _tradeoff(sigmas, q: float, rounds: int, delta: float):
    """For each sigma: REAL epsilon (subsampled-Gaussian composition) + recall."""
    out = []
    for sigma in sigmas:
        eps = dp_summary(sigma, delta, rounds, sample_rate=q)["epsilon"]
        out.append(
            {
                "sigma": _r(sigma, 3),
                "epsilon": _r(eps),
                "utility": _modelled_recall(sigma),
            }
        )
    return out


def gen_dp():
    eps_over_rounds = _epsilon_over_rounds(
        OPERATING_SIGMA, SAMPLE_RATE, NUM_ROUNDS, DELTA
    )
    operating_eps = eps_over_rounds[-1]["epsilon"]

    tradeoff = _tradeoff(TRADEOFF_SIGMAS, SAMPLE_RATE, NUM_ROUNDS, DELTA)

    out = {
        "delta": DELTA,
        "operating": {
            "sigma": _r(OPERATING_SIGMA, 3),
            "epsilon": operating_eps,
            "round": int(NUM_ROUNDS),
        },
        "epsilonOverRounds": eps_over_rounds,
        "tradeoff": tradeoff,
        "meta": {
            "note": (
                f"Real RDP accountant (Mironov 2017). Subsampled Gaussian "
                f"mechanism, Poisson sample rate q={SAMPLE_RATE}, {NUM_ROUNDS} "
                f"federated rounds, budget at delta={DELTA:.0e}. epsilonOverRounds "
                f"and every tradeoff epsilon are exact accountant outputs; the "
                f"tradeoff 'utility' (recall) is a smooth modelled curve anchored "
                f"to the federated-GNN recall regime, not a per-sigma re-train."
            ),
            "accountant": "RDP",
            "sampleRate": SAMPLE_RATE,
            "rounds": int(NUM_ROUNDS),
            "targetEpsilon": _r(TARGET_EPSILON, 3),
        },
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "dp.json")
    with open(path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    return path, out


# ── validation ─────────────────────────────────────────────────────────────
def _validate(d):
    assert set(d) == {"delta", "operating", "epsilonOverRounds", "tradeoff", "meta"}
    assert isinstance(d["delta"], (int, float)) and 0.0 < d["delta"] < 1.0

    op = d["operating"]
    assert set(op) >= {"sigma", "epsilon", "round"}
    assert op["sigma"] > 0 and op["epsilon"] > 0 and isinstance(op["round"], int)

    eor = d["epsilonOverRounds"]
    assert len(eor) >= 1
    prev_round = 0
    prev_eps = -1.0
    for pt in eor:
        assert set(pt) == {"round", "epsilon"}
        assert isinstance(pt["round"], int) and pt["round"] == prev_round + 1
        assert pt["epsilon"] >= 0.0
        # epsilon must be monotonically non-decreasing in rounds (composition).
        assert pt["epsilon"] >= prev_eps - 1e-9, "epsilon must rise with rounds"
        prev_round = pt["round"]
        prev_eps = pt["epsilon"]
    # operating epsilon == final-round epsilon
    assert abs(eor[-1]["epsilon"] - op["epsilon"]) < 1e-6

    tr = d["tradeoff"]
    assert len(tr) >= 3
    last_eps = math.inf
    for row in tr:
        assert set(row) == {"sigma", "epsilon", "utility"}
        assert row["sigma"] > 0 and row["epsilon"] >= 0.0
        assert 0.0 <= row["utility"] <= 1.0
        # ascending sigma => descending epsilon (tighter privacy).
        assert row["epsilon"] <= last_eps + 1e-9, "epsilon must fall as sigma rises"
        last_eps = row["epsilon"]

    assert d["meta"]["accountant"] == "RDP"
    assert isinstance(d["meta"]["note"], str) and d["meta"]["note"]


def main():
    print("[dp] generating dp.json — real RDP accountant ...")
    path, out = gen_dp()

    with open(path) as fh:
        _validate(json.load(fh))

    size_kb = os.path.getsize(path) / 1024.0

    print("\n==================== DP SUMMARY ====================")
    print(f"dp.json   {size_kb:6.1f} KB   {path}")
    print(
        f"operating point: sigma={out['operating']['sigma']}  "
        f"q={SAMPLE_RATE}  rounds={out['operating']['round']}  "
        f"delta={out['delta']:.0e}"
    )
    print(
        f"  -> epsilon spent = {out['operating']['epsilon']:.3f}   "
        f"(target budget epsilon = {TARGET_EPSILON})"
    )
    eor = out["epsilonOverRounds"]
    picks = [1, 5, 10, 20, 30]
    line = "  epsilon at rounds:  " + "  ".join(
        f"r{r}={eor[r - 1]['epsilon']:.3f}" for r in picks if r <= len(eor)
    )
    print(line)
    print("  noise<->utility frontier (sigma -> epsilon, recall):")
    for row in out["tradeoff"]:
        mark = "  <- operating" if abs(row["sigma"] - OPERATING_SIGMA) < 1e-6 else ""
        print(
            f"    sigma={row['sigma']:<4}  eps={row['epsilon']:7.3f}  "
            f"recall={row['utility']:.3f}{mark}"
        )
    assert size_kb < 80.0, "dp.json must stay under 80KB"
    print("====================================================")
    print(f"dp.json written ({size_kb:.1f} KB < 80KB) and schema-validated.")


if __name__ == "__main__":
    main()
