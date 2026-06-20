"""Pedersen commitments over a fixed prime-order group (reference implementation).

A Pedersen commitment binds a committer to a message ``m`` while hiding it::

    C = g^m * h^r  (mod p)

where ``r`` is a uniformly random blinding factor and ``g``, ``h`` are two
group generators whose *relative* discrete log is unknown (nobody knows ``x``
such that ``h = g^x``). The scheme is:

* **Perfectly hiding** — for any ``m``, as ``r`` ranges uniformly over the
  exponent group, ``C`` is uniformly distributed over the subgroup. A commitment
  therefore reveals nothing about ``m`` (information-theoretic hiding).
* **Computationally binding** — opening ``C`` to a different ``(m', r')`` would
  require finding ``log_g(h)``, i.e. solving a discrete log. So binding rests on
  the hardness of discrete log in this group.
* **Additively homomorphic** — ``C(m1,r1) * C(m2,r2) = g^(m1+m2) h^(r1+r2) =
  C(m1+m2, r1+r2)``. Sums of commitments commit to the sum of messages. This is
  the property the bounded-norm proof and the secure-aggregation protocol lean
  on.

The group
=========
We use the **2048-bit MODP group from RFC 3526** (the "Group 14" Oakley group).
Its modulus ``p`` is a safe prime: ``p = 2q + 1`` with ``q`` prime. The standard
generator ``2`` generates the order-``2q`` group of quadratic residues only when
squared; to get a clean **prime-order ``q`` subgroup** (so the homomorphism is
over the field ``Z_q`` of exponents, with no small-subgroup confusion) we work
in the subgroup ``QR_p`` of quadratic residues, whose order is exactly ``q``.

* ``g`` is a fixed quadratic residue (we use ``g = 4 = 2^2 mod p``, a generator
  of ``QR_p``).
* ``h`` is derived by **hash-to-group**: hash a domain-separated seed to an
  integer, reduce mod ``p``, and square it (``cand^2 mod p``) so the result lands
  in ``QR_p``. Because ``h`` comes from a hash with no known relationship to
  ``g``, its discrete log base ``g`` is unknown ("nothing-up-my-sleeve"). This is
  the standard NUMS construction and is what makes the commitment binding.

Exponents (messages ``m`` and blinds ``r``) are reduced mod ``q`` = the subgroup
order, so the additive homomorphism is exactly addition in ``Z_q``.

REFERENCE vs PRODUCTION
=======================
This pure-Python big-integer modular group is the **reference**: dependency-light
(stdlib ``hashlib`` + ``secrets`` only), auditable, and correct. **Production**
should use a fast, side-channel-resistant elliptic-curve group with a built-in
NUMS second generator — **Ristretto255** (e.g. ``curve25519-dalek`` /
``libsodium``'s ``crypto_core_ristretto255``) — which gives the same algebra at a
fraction of the cost and with constant-time operations. The API here mirrors what
an EC backend would expose so callers do not change.
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# RFC 3526, 2048-bit MODP Group ("Group 14"). p is a safe prime: p = 2q + 1.
# --------------------------------------------------------------------------- #
_P_HEX = (
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
    "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
    "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
    "DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF"
)

P = int(_P_HEX, 16)            # safe prime modulus
Q = (P - 1) // 2               # order of the quadratic-residue subgroup (prime)


def _is_qr(x: int) -> bool:
    """True iff x is a quadratic residue mod p (Euler's criterion)."""
    return pow(x % P, Q, P) == 1


# Generator of the prime-order-q subgroup QR_p. 2 is a generator of the full
# group of order 2q; its square 4 = 2^2 is a generator of the order-q subgroup
# of quadratic residues.
G = pow(2, 2, P)               # = 4
assert _is_qr(G), "G must be a quadratic residue"


def _hash_to_subgroup(label: bytes) -> int:
    """Hash ``label`` to an element of the prime-order subgroup QR_p (NUMS).

    We expand the label with counter-mode SHA-256 to a wide integer, reduce mod
    p, and square it so the result is a quadratic residue (hence in the order-q
    subgroup). We reject the trivial elements 0/1 and retry with a bumped
    counter. The discrete log of the output base ``G`` is unknown because the
    output is determined solely by the hash.
    """
    counter = 0
    while True:
        buf = b""
        for i in range(16):  # 16 * 32 bytes = 4096 bits >> 2048, no modulo bias issues of note
            buf += hashlib.sha256(label + counter.to_bytes(4, "big") + i.to_bytes(4, "big")).digest()
        cand = int.from_bytes(buf, "big") % P
        h = pow(cand, 2, P)  # force into QR_p (order q)
        if h not in (0, 1):
            return h
        counter += 1


# Second generator h. Nothing-up-my-sleeve: derived purely from a fixed,
# documented domain-separation label, so log_G(H) is unknown.
H = _hash_to_subgroup(b"veritas/pedersen/h/rfc3526-group14/v1")
assert _is_qr(H), "H must be a quadratic residue"


@dataclass(frozen=True)
class Commitment:
    """A Pedersen commitment value ``C`` in QR_p (an integer mod p).

    Carries the group parameters it was made under so verification and the
    homomorphic combine can sanity-check compatibility.
    """

    C: int
    p: int = P
    g: int = G
    h: int = H

    def __mul__(self, other: "Commitment") -> "Commitment":
        """Homomorphic combine: C(m1,r1) * C(m2,r2) = C(m1+m2, r1+r2)."""
        if not isinstance(other, Commitment):
            return NotImplemented
        if (self.p, self.g, self.h) != (other.p, other.g, other.h):
            raise ValueError("cannot combine commitments from different groups")
        return Commitment((self.C * other.C) % self.p, self.p, self.g, self.h)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Commitment):
            return NotImplemented
        return (self.C, self.p, self.g, self.h) == (other.C, other.p, other.g, other.h)

    def __hash__(self) -> int:
        return hash((self.C, self.p, self.g, self.h))

    def to_bytes(self) -> bytes:
        """Canonical byte encoding of C (for Fiat-Shamir transcripts)."""
        return self.C.to_bytes((self.p.bit_length() + 7) // 8, "big")


def random_blind() -> int:
    """Sample a uniformly random blinding factor r in [0, q)."""
    return secrets.randbelow(Q)


def commit(m: int, r: int | None = None) -> tuple[Commitment, int]:
    """Commit to integer ``m`` with blind ``r`` (sampled if ``None``).

    Returns ``(Commitment, r)`` so the caller can keep the opening ``r``.
    Exponents are reduced mod q, so the homomorphism is addition in Z_q.
    """
    if r is None:
        r = random_blind()
    m_mod = m % Q
    r_mod = r % Q
    C = (pow(G, m_mod, P) * pow(H, r_mod, P)) % P
    return Commitment(C), r


def commit_value(m: int, r: int) -> Commitment:
    """Commit and return only the Commitment (r supplied by caller)."""
    c, _ = commit(m, r)
    return c


def verify_open(C: Commitment, m: int, r: int) -> bool:
    """Check that ``C`` opens to ``(m, r)``: C == g^m h^r (mod p)."""
    if not isinstance(C, Commitment):
        return False
    expected = (pow(C.g, m % Q, C.p) * pow(C.h, r % Q, C.p)) % C.p
    return C.C == expected


def add(c1: Commitment, c2: Commitment) -> Commitment:
    """Homomorphic addition of two commitments (functional form of ``*``)."""
    return c1 * c2
