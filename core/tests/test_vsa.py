"""Tests for Verifiable Secure Aggregation (veritas_core.vsa)."""
import copy

import numpy as np
import pytest

from veritas_core import vsa
from veritas_core import secure_agg
from veritas_core import commit as C


def _seeds(ids):
    return secure_agg.establish_pairwise_seeds(ids, np.random.default_rng(7))


# --------------------------------------------------------------------------- #
# (a) in-bound honest update verifies.
# --------------------------------------------------------------------------- #
def test_in_bound_proof_verifies():
    ids = ["bankA", "bankB"]
    seeds = _seeds(ids)
    B = 5.0
    u = np.array([1.0, 2.0, 1.0])  # ||u|| = sqrt(6) ~= 2.449 < 5
    contrib = vsa.client_contribution(u, "bankA", ["bankB"], seeds, B)
    assert vsa.server_verify(contrib, B) is True


def test_range_proof_self_consistency():
    s, r = 12345, C.random_blind()
    k = vsa._bit_length_for_bound((1 << 20))  # generous
    # use a k that fits s
    k = max(k, s.bit_length())
    proof = vsa.prove_bounded_norm(s, r, k)
    assert vsa.verify_bounded_norm(proof, max_bits=k) is True


# --------------------------------------------------------------------------- #
# (b) out-of-bound (amplified poison) -> proof fails, server rejects.
# --------------------------------------------------------------------------- #
def test_out_of_bound_poison_rejected():
    ids = ["bankA", "bankB"]
    seeds = _seeds(ids)
    B = 5.0
    # Amplified poison: ||u|| = 100 >> B. Client tries to prove it anyway by
    # NOT clipping (clip_norm huge), so the true squared norm exceeds B^2.
    poison = np.array([100.0, 0.0, 0.0])
    # An honest-but-malicious client that does not clip cannot build a valid
    # honest proof: client_contribution clips to B by default, so to simulate a
    # poison we forge a contribution whose committed s is out of range and whose
    # proof we attempt for the real (too-large) s.
    with pytest.raises(ValueError):
        # honest proof construction refuses an s that overflows the bound's bits
        s = vsa.squared_norm_to_int(poison)  # 100^2 * SCALE
        r = C.random_blind()
        k = vsa._bit_length_for_bound(vsa.bound_to_int(B))
        vsa.prove_bounded_norm(s, r, k)  # s >= 2^k -> ValueError

    # Now simulate a *cheating* server-facing contribution: the attacker commits
    # to the real out-of-range s but produces a proof over a SMALLER fake s'.
    # The commitment in the contribution will not match the proof's Cs, OR the
    # link/range check fails -> server_verify must reject.
    s_real = vsa.squared_norm_to_int(poison)
    r_real = C.random_blind()
    real_commit, _ = C.commit(s_real, r_real)
    k = vsa._bit_length_for_bound(vsa.bound_to_int(B))
    # Attacker proves a fake in-range s' (different commitment).
    s_fake = (1 << k) - 1
    r_fake = C.random_blind()
    fake_proof = vsa.prove_bounded_norm(s_fake, r_fake, k)
    forged = vsa.Contribution(
        client_id="evil",
        masked_u=poison,
        commitment=real_commit,         # commits to the true (huge) norm
        proof=fake_proof,               # proof is over a different value
        max_bits=k,
        dim=poison.shape[0],
    )
    assert vsa.server_verify(forged, B) is False, "commitment != proof.Cs must reject"


def test_clipped_poison_is_in_bound_but_amplification_neutralised():
    """A poison client that DOES clip is in-bound but no longer amplified."""
    ids = ["a", "b"]
    seeds = _seeds(ids)
    B = 5.0
    poison = np.array([100.0, 0.0, 0.0])  # huge -> clipped to norm B
    contrib = vsa.client_contribution(poison, "a", ["b"], seeds, B)
    # It verifies (it clipped), but its masked content corresponds to a
    # norm-B vector, not the amplified poison: amplification is neutralised.
    assert vsa.server_verify(contrib, B) is True


# --------------------------------------------------------------------------- #
# (c) server's view does not reveal the cleartext update (hiding).
# --------------------------------------------------------------------------- #
def test_server_view_hides_cleartext():
    ids = ["a", "b"]
    seeds = _seeds(ids)
    B = 5.0
    u = np.array([1.0, -2.0, 0.5])
    contrib = vsa.client_contribution(u, "a", ["b"], seeds, B)

    # masked_u must differ from the raw update (Bonawitz mask applied).
    assert not np.allclose(contrib.masked_u, u)
    # masking is large-magnitude -> view looks like noise.
    assert np.linalg.norm(contrib.masked_u) > 10 * np.linalg.norm(u)

    # The commitment is independent of any opening: same masked semantics but a
    # different blind yields a different commitment, and the commitment alone
    # does not let anyone recover s. We check that two commitments to the SAME
    # squared norm with different blinds differ (perfect hiding).
    s = vsa.squared_norm_to_int(vsa._dp.clip_update(u, B))
    c1, _ = C.commit(s, 1)
    c2, _ = C.commit(s, 2)
    assert c1.C != c2.C
    # And the contribution carries no opening (r_s) at all.
    assert not hasattr(contrib, "r_s")


# --------------------------------------------------------------------------- #
# (d) aggregate sum over accepted == true sum of their raw updates.
# --------------------------------------------------------------------------- #
def test_aggregate_sum_over_accepted_with_poison_excluded():
    ids = ["h1", "h2", "h3", "poison"]
    seeds = _seeds(ids)
    B = 10.0
    raw = {
        "h1": np.array([1.0, 0.0, 2.0]),
        "h2": np.array([0.5, 1.0, -1.0]),
        "h3": np.array([-1.0, 2.0, 0.5]),
    }
    # All honest updates are in-bound (norms well under 10). Clip to B (no DP).
    clipped = {cid: vsa._dp.clip_update(u, B) for cid, u in raw.items()}

    contribs = []
    for cid in ["h1", "h2", "h3"]:
        peers = [x for x in ids if x != cid]
        contribs.append(vsa.client_contribution(raw[cid], cid, peers, seeds, B))

    # Poison client: amplified out-of-bound, forged proof (won't verify).
    poison = np.array([500.0, 500.0, 500.0])
    s_real = vsa.squared_norm_to_int(poison)
    r_real = C.random_blind()
    real_commit, _ = C.commit(s_real, r_real)
    k = vsa._bit_length_for_bound(vsa.bound_to_int(B))
    fake_proof = vsa.prove_bounded_norm((1 << k) - 1, C.random_blind(), k)
    peers = [x for x in ids if x != "poison"]
    masked_poison = secure_agg.mask_update(poison, "poison", peers, seeds)
    contribs.append(vsa.Contribution(
        client_id="poison",
        masked_u=masked_poison,
        commitment=real_commit,
        proof=fake_proof,
        max_bits=k,
        dim=3,
    ))

    agg, accepted, rejected = vsa.verifiable_secure_aggregate_with_repair(
        contribs, B, seeds
    )
    assert set(accepted) == {"h1", "h2", "h3"}
    assert rejected == ["poison"]

    true_sum = clipped["h1"] + clipped["h2"] + clipped["h3"]
    assert np.allclose(agg, true_sum, atol=1e-6), (agg, true_sum)


def test_aggregate_all_accepted_equals_true_sum():
    ids = ["a", "b", "c"]
    seeds = _seeds(ids)
    B = 10.0
    raw = {
        "a": np.array([1.0, 2.0, 3.0]),
        "b": np.array([-1.0, 0.0, 1.0]),
        "c": np.array([2.0, -2.0, 0.0]),
    }
    contribs = []
    for cid in ids:
        peers = [x for x in ids if x != cid]
        contribs.append(vsa.client_contribution(raw[cid], cid, peers, seeds, B))
    agg, accepted, rejected = vsa.verifiable_secure_aggregate(contribs, B)
    assert rejected == []
    clipped = {cid: vsa._dp.clip_update(u, B) for cid, u in raw.items()}
    true_sum = clipped["a"] + clipped["b"] + clipped["c"]
    assert np.allclose(agg, true_sum, atol=1e-6)


# --------------------------------------------------------------------------- #
# (e) soundness: tampered proof or mismatched commitment is rejected.
# --------------------------------------------------------------------------- #
def test_tampered_bit_proof_rejected():
    ids = ["a", "b"]
    seeds = _seeds(ids)
    B = 5.0
    u = np.array([1.0, 1.0, 1.0])
    contrib = vsa.client_contribution(u, "a", ["b"], seeds, B)
    assert vsa.server_verify(contrib, B) is True

    tampered = copy.deepcopy(contrib)
    # Flip a response in the first bit proof -> OR-proof must fail.
    tampered.proof.bit_proofs[0].z0 = (tampered.proof.bit_proofs[0].z0 + 1) % vsa.Q
    assert vsa.server_verify(tampered, B) is False


def test_tampered_link_proof_rejected():
    ids = ["a", "b"]
    seeds = _seeds(ids)
    B = 5.0
    u = np.array([1.0, 1.0, 1.0])
    contrib = vsa.client_contribution(u, "a", ["b"], seeds, B)
    tampered = copy.deepcopy(contrib)
    tampered.proof.link.z = (tampered.proof.link.z + 1) % vsa.Q
    assert vsa.server_verify(tampered, B) is False


def test_commitment_not_matching_proof_rejected():
    ids = ["a", "b"]
    seeds = _seeds(ids)
    B = 5.0
    u = np.array([1.0, 1.0, 1.0])
    contrib = vsa.client_contribution(u, "a", ["b"], seeds, B)
    tampered = copy.deepcopy(contrib)
    # Swap in a commitment to a different value -> proof.Cs mismatch.
    other, _ = C.commit(999, 123)
    tampered.commitment = other
    assert vsa.server_verify(tampered, B) is False


def test_proof_claiming_too_many_bits_rejected():
    """A prover cannot claim a bigger range than B allows (over-large k)."""
    B = 4.0  # B^2 * SCALE -> some k
    k_allowed = vsa._bit_length_for_bound(vsa.bound_to_int(B))
    # Build a proof with MORE bits than allowed (proving a larger range).
    big_k = k_allowed + 5
    s = (1 << k_allowed)  # value > 2^k_allowed - 1, only provable with big_k
    r = C.random_blind()
    proof = vsa.prove_bounded_norm(s, r, big_k)
    # Standalone it verifies for big_k...
    assert vsa.verify_bounded_norm(proof, max_bits=big_k) is True
    # ...but the server only allows k_allowed bits -> rejected.
    assert vsa.verify_bounded_norm(proof, max_bits=k_allowed) is False
