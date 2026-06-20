"""Bonawitz-style additive pairwise-masking secure aggregation.

This module implements the canonical *secure sum* protocol used to let an
aggregator (in Veritas: the bank node receiving Tier-0 device updates) recover
the SUM of many client updates while **never** seeing any individual client's
update in the clear. It replaces the v1 posture where ``edge_aggregate`` merely
DP-noised each device update and folded it in one-at-a-time (the bank still saw
every individual, noised vector).

The protocol (Bonawitz et al., "Practical Secure Aggregation for
Privacy-Preserving Machine Learning", CCS 2017)
=================================================================

Pairwise masking that cancels
-----------------------------
For every ordered pair of clients (i, j) with i < j there is a shared secret
seed ``s_ij``. From that seed both clients deterministically derive the SAME
pseudo-random mask vector ``m_ij = PRG(s_ij)``. They apply it with OPPOSITE
signs determined purely by id ordering:

    client i (the lower id)  ADDS     m_ij
    client j (the higher id) SUBTRACTS m_ij

So client i sends ``x_i + Σ_{j>i} m_ij - Σ_{j<i} m_ji`` (raw update plus a sum
of masks). Each mask ``m_ij`` appears exactly twice across the cohort: ``+m_ij``
in client i's message and ``-m_ij`` in client j's message. When the server
**adds up all masked messages**, every pairwise mask cancels and the server is
left with exactly ``Σ_i x_i`` — the true sum — having observed only individual
vectors that are uniformly-random-looking (each is its real update buried under
a sum of large random masks). This is the security property: an honest-but-curious
server learns NOTHING about any single client's update, only the aggregate.

Where the shared seeds come from (production vs. this reference)
---------------------------------------------------------------
PRODUCTION: ``s_ij`` is established by an **authenticated X25519
Diffie-Hellman** exchange. Each client publishes an X25519 public key (signed by
an identity key so the server cannot mount a man-in-the-middle). Client i and
client j independently compute the same DH shared secret
``s_ij = HKDF(X25519(sk_i, pk_j)) == HKDF(X25519(sk_j, pk_i))``. The server only
relays public keys; it never learns any ``s_ij`` and therefore can never derive
the masks to peel them off an individual message. The server is structurally
incapable of unmasking a single client.

THIS REFERENCE: to keep the core pure-numpy and dependency-free (no
``cryptography``/X25519 library), ``establish_pairwise_seeds`` uses a **trusted
dealer** (a local RNG) to hand out the seeds. This stands in for the DH step and
exercises the exact same masking algebra; only the seed-establishment channel
differs. The trusted dealer also lets the reference reconstruct dropped clients'
seeds for the dropout path (see below) without implementing Shamir sharing.

Dropout robustness (the hard part Bonawitz solves)
--------------------------------------------------
If client j masks against client i (using ``m_ij``) but then DROPS OUT before
its own masked message reaches the server, ``m_ij`` is now applied only ONCE
(by the survivor) and no longer cancels — the recovered sum is corrupted by a
large random vector. Bonawitz fixes this by having every client **Shamir
secret-share** the material needed to reconstruct its pairwise masks among the
others. After a dropout, the surviving clients send their shares of the dropped
client's seeds to the server; with a threshold ``t`` of shares the server
reconstructs exactly the masks that failed to cancel and subtracts them,
recovering the correct sum **over the surviving clients only**. Crucially the
threshold prevents the server from reconstructing a *present* client's secrets.

THIS REFERENCE: ``recover_dropout`` takes the dropped clients' pairwise seeds
(which, in production, the server reconstructs via Shamir from surviving shares;
here they come from the trusted dealer's seed table) and subtracts the
now-uncancelled masks so ``secure_sum`` again yields the true sum over the
survivors.

Public API
==========
- ``derive_pairwise_mask(seed_bytes, dim) -> np.ndarray``  — the PRG.
- ``establish_pairwise_seeds(client_ids, master_rng=None) -> dict``  — dealer.
- ``mask_update(update, client_id, peer_ids, shared_seeds) -> np.ndarray``.
- ``secure_sum(masked_updates) -> np.ndarray``  — the SERVER's only operation.
- ``recover_dropout(running_sum, dropped_ids, survivor_ids, seed_table, dim)``.
- ``secure_aggregate(updates, client_ids, *, seed_table=None, dp=None) -> sum``.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Mask magnitude. Masks must dominate the real update so that an individual
# masked vector reveals nothing — a curious server staring at one message sees
# essentially uniform noise. The masks cancel exactly regardless of scale, so we
# pick a large spread.
_MASK_SCALE = 1.0e6

# Canonical seed length: 32 raw bytes (256 bits), the HMAC-SHA256 key size.
SEED_BYTES = 32

SeedKey = Tuple[str, str]  # canonical (low_id, high_id)


def _canon(a: str, b: str) -> SeedKey:
    """Canonical ordering of a client-id pair (lower id first)."""
    return (a, b) if a < b else (b, a)


# --------------------------------------------------------------------------- #
# Canonical, byte-identical cross-language PRG (SPEC-PRG).
#
# This is THE one definition of the pairwise-mask randomness. The edge-sdk
# (edge-sdk/src/secureAgg.ts) and the node re-derive the SAME floats from the
# SAME 32-byte seed, so a TS-masked update and a Python ``secure_sum`` actually
# interoperate — the masks cancel because the bytes are identical, not merely
# "the algebra is similar".
#
# Algorithm (HMAC-SHA256 in counter mode):
#   for counter c = 0, 1, 2, ...:
#     block = HMAC_SHA256(key=seed, msg=c.to_bytes(8, "big"))   # 32 bytes
#     split block into four 8-byte little-endian chunks -> uint64
#     uniform = uint64 / 2**64   in [0, 1)
#   take the first n uniforms.
# --------------------------------------------------------------------------- #
def seed_to_hex(seed_bytes: bytes) -> str:
    """Represent a 32-byte seed on the wire as 64-char lowercase hex."""
    if len(seed_bytes) != SEED_BYTES:
        raise ValueError(f"seed must be {SEED_BYTES} bytes, got {len(seed_bytes)}")
    return seed_bytes.hex()


def hex_to_seed(seed_hex: str) -> bytes:
    """Decode a 64-char lowercase hex seed back to 32 raw bytes."""
    b = bytes.fromhex(seed_hex)
    if len(b) != SEED_BYTES:
        raise ValueError(f"seed hex must decode to {SEED_BYTES} bytes")
    return b


def prg_floats(seed: bytes, n: int) -> np.ndarray:
    """Expand a 32-byte seed into ``n`` uniform float64s in [0, 1).

    HMAC-SHA256 counter mode (see module note). Deterministic and byte-identical
    to the TypeScript ``prgFloats`` in edge-sdk/src/secureAgg.ts — a golden test
    vector pins this down across both languages.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    out = np.empty(n, dtype=np.float64)
    filled = 0
    counter = 0
    inv = float(2 ** 64)
    while filled < n:
        block = hmac.new(seed, counter.to_bytes(8, "big"), hashlib.sha256).digest()
        for j in range(4):
            if filled >= n:
                break
            val = int.from_bytes(block[j * 8 : (j + 1) * 8], "little")
            out[filled] = val / inv
            filled += 1
        counter += 1
    return out


def derive_pairwise_mask(seed_bytes: bytes, dim: int) -> np.ndarray:
    """PRG: expand a shared secret seed into a deterministic mask vector.

    Both clients in a pair derive the identical vector from the identical seed.
    The uniforms come from the canonical byte-identical :func:`prg_floats`
    (HMAC-SHA256 counter mode), then are mapped onto the zero-centred
    ``[-_MASK_SCALE, _MASK_SCALE)`` range so the mask dominates any realistic
    update. Deterministic in ``seed_bytes`` and ``dim``, and byte-for-byte
    reproducible in the edge-sdk / node.

    ``seed_bytes`` should be 32 raw bytes; shorter/other seeds are hashed to 32
    bytes first so any caller-supplied seed string still maps to a valid key.
    """
    if dim <= 0:
        raise ValueError("dim must be positive")
    if len(seed_bytes) != SEED_BYTES:
        seed_bytes = hashlib.sha256(seed_bytes).digest()
    u = prg_floats(seed_bytes, dim)
    # Uniform in [0,1) -> zero-centred, large magnitude.
    return (u * 2.0 - 1.0) * _MASK_SCALE


def establish_pairwise_seeds(
    client_ids: Sequence[str], master_rng: Optional[np.random.Generator] = None
) -> Dict[SeedKey, bytes]:
    """Trusted-dealer stand-in for the X25519 DH key-agreement step.

    Returns a table mapping each canonical (low_id, high_id) pair to a fresh
    random 32-byte seed ``s_ij``. In production this table is NEVER materialised
    on the server: each ``s_ij`` is derived independently by clients i and j from
    an authenticated Diffie-Hellman exchange, so the server cannot learn it.
    Here a local RNG (or ``secrets`` when no rng is given) plays the dealer so
    the reference/test path can exercise the masking algebra end-to-end.
    """
    ids = list(client_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("client_ids must be unique")
    seeds: Dict[SeedKey, bytes] = {}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            key = _canon(ids[a], ids[b])
            if master_rng is None:
                seeds[key] = secrets.token_bytes(32)
            else:
                seeds[key] = master_rng.integers(
                    0, 256, size=32, dtype=np.uint8
                ).tobytes()
    return seeds


def _seed_for(shared_seeds: Dict[SeedKey, bytes], a: str, b: str) -> bytes:
    key = _canon(a, b)
    if key not in shared_seeds:
        raise KeyError(f"no shared seed for pair {key}")
    return shared_seeds[key]


def mask_update(
    update: np.ndarray,
    client_id: str,
    peer_ids: Iterable[str],
    shared_seeds: Dict[SeedKey, bytes],
) -> np.ndarray:
    """Apply this client's pairwise masks to its raw update.

    For each peer ``p``: derive ``m = PRG(s_{client,p})`` and ADD it if
    ``client_id < p`` else SUBTRACT it. The sign is purely a function of id
    ordering, so the two clients in a pair always apply the same mask with
    opposite signs — guaranteeing cancellation in ``secure_sum``.

    Returns ``update + Σ_p sign(p) * m_{client,p}`` — what the client sends to
    the server. The raw ``update`` never leaves the client.
    """
    u = np.asarray(update, dtype=np.float64)
    masked = u.copy()
    for p in peer_ids:
        if p == client_id:
            continue
        seed = _seed_for(shared_seeds, client_id, p)
        m = derive_pairwise_mask(seed, u.shape[0])
        sign = 1.0 if client_id < p else -1.0
        masked = masked + sign * m
    return masked


def secure_sum(masked_updates: Sequence[np.ndarray]) -> np.ndarray:
    """The SERVER's entire view and operation: elementwise sum of masked vectors.

    With all clients present, every pairwise mask appears once with ``+`` and
    once with ``-``, so they cancel and this equals ``Σ_i x_i`` exactly (up to
    floating point). The server never sees, stores, or can derive any individual
    ``x_i`` — only the masked messages and their sum.
    """
    if not masked_updates:
        raise ValueError("no masked updates to sum")
    acc = np.zeros_like(np.asarray(masked_updates[0], dtype=np.float64))
    for mv in masked_updates:
        acc = acc + np.asarray(mv, dtype=np.float64)
    return acc


def _uncancelled_mask_for_dropout(
    dropped_id: str,
    survivor_ids: Sequence[str],
    seed_table: Dict[SeedKey, bytes],
    dim: int,
) -> np.ndarray:
    """Net mask contributed by survivors against ONE dropped client.

    When ``dropped_id`` disappears, each surviving peer ``s`` still added
    ``sign(s, dropped) * m_{s,dropped}`` to its own message, but the dropped
    client's cancelling ``-sign`` term is missing. The leftover in the running
    sum is therefore ``Σ_s sign(s,dropped) * m_{s,dropped}``. We recompute that
    so the caller can subtract it back out.
    """
    leftover = np.zeros(dim, dtype=np.float64)
    for s in survivor_ids:
        if s == dropped_id:
            continue
        seed = _seed_for(seed_table, s, dropped_id)
        m = derive_pairwise_mask(seed, dim)
        # From the SURVIVOR's perspective: it added sign(survivor < dropped).
        sign = 1.0 if s < dropped_id else -1.0
        leftover = leftover + sign * m
    return leftover


def recover_dropout(
    running_sum: np.ndarray,
    dropped_ids: Sequence[str],
    survivor_ids: Sequence[str],
    seed_table: Dict[SeedKey, bytes],
    dim: Optional[int] = None,
) -> np.ndarray:
    """Repair ``secure_sum`` after one or more clients drop out post-masking.

    ``running_sum`` is ``secure_sum`` over the surviving clients' masked
    messages. Because the dropped clients never submitted their cancelling
    terms, ``running_sum`` carries leftover masks. Given the dropped clients'
    pairwise seeds (production: reconstructed via Shamir from surviving shares;
    reference: from ``seed_table``), we subtract each leftover so the result is
    exactly ``Σ_{survivors} x_i``.

    Returns the corrected sum over the surviving clients only.
    """
    rs = np.asarray(running_sum, dtype=np.float64)
    d = rs.shape[0] if dim is None else dim
    corrected = rs.copy()
    for dropped in dropped_ids:
        corrected = corrected - _uncancelled_mask_for_dropout(
            dropped, survivor_ids, seed_table, d
        )
    return corrected


def secure_aggregate(
    updates: Sequence[np.ndarray],
    client_ids: Sequence[str],
    *,
    seed_table: Optional[Dict[SeedKey, bytes]] = None,
    dp=None,
) -> np.ndarray:
    """Convenience: establish seeds, mask every update, and return the secure sum.

    Each client is expected to have already clipped + DP-noised its update before
    masking (``dp`` here is purely informational — the privacy is applied
    client-side in ``updates``, so the recovered sum already reflects it). The
    masking is what makes the *aggregator's view* private; DP makes the *output*
    private. The two compose: clip+noise, then mask, then sum.

    Returns ``Σ_i updates[i]`` while having only ever exposed masked individuals
    to the (notional) server.
    """
    if len(updates) != len(client_ids):
        raise ValueError("updates and client_ids length mismatch")
    if seed_table is None:
        seed_table = establish_pairwise_seeds(client_ids)
    masked: List[np.ndarray] = []
    for cid, u in zip(client_ids, updates):
        peers = [c for c in client_ids if c != cid]
        masked.append(mask_update(u, cid, peers, seed_table))
    return secure_sum(masked)
