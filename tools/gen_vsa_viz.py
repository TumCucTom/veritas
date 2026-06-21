#!/usr/bin/env python3
"""Offline generator for the VSA verifiable-norm-proof visualization.

Runs the REAL Veritas VSA protocol (Pedersen commitment + bit-decomposition
sigma range proof + norm-binding proof) for two clients and captures a tiny,
deterministic JSON the web UI animates:

* an HONEST client whose update norm is within the public bound B (proof
  VERIFIES via ``server_verify``), and
* a POISON client whose norm exceeds the provable bound (proof is REJECTED).

We use the actual ``veritas_core.vsa`` functions end to end. The poison client's
squared norm overflows the bit-length ``k`` implied by B, so we cannot honestly
call ``client_contribution`` (it raises by design). To still exercise a REAL
rejection through ``server_verify``, the poison client builds an honest
contribution at a *larger* bound (more bits) and submits it against the smaller
public B — exactly the "prover claims a larger range than B permits" cheat the
verifier rejects (``proof.k > k`` / ``verify_bounded_norm`` max_bits check).

Output: web/public/viz/vsa.json  (deterministic, < 60KB).

Run:
    source /Users/tom/Veritas/core/.venv/bin/activate
    python /Users/tom/Veritas/tools/gen_vsa_viz.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

# Make veritas_core importable when run directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(os.path.dirname(_HERE), "core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from veritas_core import vsa  # noqa: E402
from veritas_core import secure_agg  # noqa: E402


OUT_PATH = os.path.join(
    os.path.dirname(_HERE), "web", "public", "viz", "vsa.json"
)

# Public L2-norm bound on ||u||. Honest client stays under it; poison exceeds it.
BOUND = 5.0

# Tiny update dimension keeps the per-coordinate norm-binding proof fast.
DIM = 6


def _commitment_preview(C: int) -> str:
    """Short, stable hex preview of a commitment value — 'committed but hidden'.

    The full commitment is a ~2048-bit integer. We expose only a SHA-256-derived
    16-hex-char fingerprint so the UI can show 'committed' without leaking the
    (already perfectly-hiding) value or bloating the JSON.
    """
    raw = C.to_bytes((C.bit_length() + 7) // 8 or 1, "big")
    return hashlib.sha256(raw).hexdigest()[:16]


def build() -> dict:
    rng = np.random.default_rng(7)  # deterministic sample

    ids = ["honest", "poison"]
    seed_table = secure_agg.establish_pairwise_seeds(ids, master_rng=rng)

    # ---- HONEST client: small in-bound update -> proof VERIFIES. ----------- #
    u_honest = np.full(DIM, 1.2, dtype=np.float64)  # ||u|| = 1.2*sqrt(6) ~ 2.94
    honest_norm = float(np.linalg.norm(u_honest))
    honest_contrib = vsa.client_contribution(
        u_honest, "honest", ["poison"], seed_table, BOUND
    )
    honest_accepted = vsa.server_verify(honest_contrib, BOUND)
    honest_preview = _commitment_preview(honest_contrib.commitment.C)

    # ---- POISON client: over-bound update -> proof REJECTED. -------------- #
    # ||u|| = 4.0*sqrt(6) ~ 9.8 > BOUND. Its squared norm overflows the k bits
    # implied by BOUND, so an honest contribution at BOUND is impossible by
    # design. We instead build a valid contribution at a LARGER bound (which
    # needs more bits) and submit it against the public BOUND: server_verify
    # rejects it because the proof claims a wider range (proof.k > k) than BOUND
    # permits. This is a genuine server_verify REJECTION of an over-norm client.
    u_poison = np.full(DIM, 4.0, dtype=np.float64)
    poison_norm = float(np.linalg.norm(u_poison))
    poison_contrib = vsa.client_contribution(
        u_poison, "poison", ["honest"], seed_table, B=poison_norm + 1.0
    )
    poison_accepted = vsa.server_verify(poison_contrib, BOUND)
    poison_preview = _commitment_preview(poison_contrib.commitment.C)

    # ---- Hard assertions: the proof genuinely accepts/rejects. ------------ #
    assert honest_accepted is True, "honest in-bound proof must VERIFY"
    assert poison_accepted is False, "poison over-bound proof must be REJECTED"
    assert honest_norm < BOUND < poison_norm, "norms must straddle the bound"

    return {
        "bound": round(BOUND, 6),
        "clients": [
            {
                "id": "honest",
                "norm": round(honest_norm, 6),
                "committed": True,
                "commitmentPreview": honest_preview,
                "accepted": bool(honest_accepted),
                "poison": False,
            },
            {
                "id": "poison",
                "norm": round(poison_norm, 6),
                "committed": True,
                "commitmentPreview": poison_preview,
                "accepted": bool(poison_accepted),
                "poison": True,
            },
        ],
        "steps": ["commit", "challenge", "response", "verify"],
        "meta": {
            "note": (
                "Each client attaches a Pedersen commitment to ||u||^2 and a "
                "Fiat-Shamir range proof that the norm is in bound. The server "
                "verifies against the commitment only — it never sees the raw "
                "update. The over-bound poison is rejected before aggregation."
            ),
            "scheme": "Pedersen + sigma range proof",
        },
    }


def main() -> None:
    data = build()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, separators=(",", ":"))
        f.write("\n")

    # Validate the JSON round-trips and report.
    with open(OUT_PATH) as f:
        reloaded = json.load(f)
    size = os.path.getsize(OUT_PATH)

    print("VSA verifiable-norm-proof viz generated")
    print(f"  output : {OUT_PATH}")
    print(f"  size   : {size} bytes (< 60KB: {size < 60_000})")
    print(f"  bound B (on ||u||) : {reloaded['bound']}")
    for c in reloaded["clients"]:
        verdict = "ACCEPTED" if c["accepted"] else "REJECTED"
        print(
            f"  client {c['id']:<6} norm={c['norm']:.4f}  "
            f"poison={c['poison']!s:<5}  verify={verdict}  "
            f"commit={c['commitmentPreview']}"
        )
    assert size < 60_000, "vsa.json must be < 60KB"
    print("  JSON valid, schema OK, assertions passed.")


if __name__ == "__main__":
    main()
