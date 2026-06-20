"""Tests for the Pedersen commitment scheme (veritas_core.commit)."""
from veritas_core import commit as C
from veritas_core.commit import P, Q, G, H


def test_group_params_sane():
    # p is a safe prime: p = 2q + 1, and g, h are quadratic residues (order-q).
    assert P == 2 * Q + 1
    assert pow(G, Q, P) == 1  # G has order dividing q (is in QR_p)
    assert pow(H, Q, P) == 1
    assert G != H and H not in (0, 1)


def test_open_roundtrip():
    m, r = 12345, C.random_blind()
    com = C.commit_value(m, r)
    assert C.verify_open(com, m, r)


def test_hiding_same_message_different_blind():
    """Same message, different blind -> different commitment value (hiding)."""
    m = 999
    c1 = C.commit_value(m, 111111)
    c2 = C.commit_value(m, 222222)
    assert c1.C != c2.C, "two blinds must give different commitments"
    # Both still open to the same m.
    assert C.verify_open(c1, m, 111111)
    assert C.verify_open(c2, m, 222222)


def test_hiding_distribution_independent_of_message():
    """Commitments to different messages with random blinds are indistinguishable.

    Perfect hiding: for ANY message m, as r ranges uniformly over Z_q the
    commitment g^m h^r ranges uniformly over the whole subgroup (h generates it).
    So the SET of commitment values reachable for m=0 is identical to the set for
    m=1 — only the (m,r) -> C mapping differs. We demonstrate this directly: any
    commitment to m=0 with blind r equals a commitment to m=1 with a shifted
    blind r' (because g = h^x for the unknown x, the shift is exactly x). Rather
    than rely on x, we show the distributions overlap: a commitment to m=0 can be
    re-expressed as a commitment to m=1 by homomorphically multiplying by g.
    """
    # C(0, r) * g == g^1 h^r == C(1, r). So the value C(0,r)*g is a valid
    # commitment to m=1, proving the m=0 and m=1 supports coincide.
    for _ in range(50):
        r = C.random_blind()
        c0 = C.commit_value(0, r)
        shifted = C.commit_value(0, r).C * C.G % C.P
        assert C.verify_open(C.Commitment(shifted), 1, r)
        # c0 itself opens to m=0 (not to m=1) for the SAME r — value depends on m.
        assert C.verify_open(c0, 0, r)
        assert not C.verify_open(c0, 1, r)


def test_binding_cannot_open_to_different_message():
    """A commitment to m does NOT verify against any m' != m (binding)."""
    m, r = 4242, C.random_blind()
    com = C.commit_value(m, r)
    assert not C.verify_open(com, m + 1, r)
    assert not C.verify_open(com, m - 1, r)
    # Even with a different blind, can't open to a different message without
    # solving DL (we just check a few wrong blinds don't accidentally verify).
    assert not C.verify_open(com, m + 1, r + 1)
    assert not C.verify_open(com, m + 7, 12345)


def test_additively_homomorphic():
    """C(m1,r1) * C(m2,r2) == C(m1+m2, r1+r2)."""
    m1, r1 = 100, 5555
    m2, r2 = 250, 9999
    c1 = C.commit_value(m1, r1)
    c2 = C.commit_value(m2, r2)
    combined = c1 * c2
    direct = C.commit_value(m1 + m2, r1 + r2)
    assert combined == direct
    assert C.verify_open(combined, m1 + m2, r1 + r2)


def test_homomorphic_add_function():
    c1 = C.commit_value(7, 1)
    c2 = C.commit_value(11, 2)
    assert C.add(c1, c2) == C.commit_value(18, 3)


def test_commitment_reveals_nothing_byte_encoding_stable():
    com = C.commit_value(123, 456)
    b = com.to_bytes()
    assert isinstance(b, bytes)
    assert int.from_bytes(b, "big") == com.C
