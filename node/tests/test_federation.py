"""Federation round against the FAKE in-memory plane.

enrol -> train -> submit DP update (auth'd with EdDSA JWT) -> receive new model.
No live control plane, no network.
"""
import numpy as np

from veritas_core.data import make_bank_data

from node.attestation import SoftwareAttestor
from node.federation import FederationClient
from node.identity import NodeIdentity

from .fake_plane import FakePlane


def _client(plane, member_id="node0", tenant="tenant0"):
    ident = NodeIdentity.generate(member_id, tenant)
    attestor = SoftwareAttestor(ident, image_id="test", config_digest="test")
    return FederationClient(ident, plane, attestor, rng=np.random.default_rng(1))


def test_enroll_registers_public_key():
    plane = FakePlane(min_updates=1)
    client = _client(plane)
    resp = client.enroll()
    assert resp["status"] == "active"
    assert "node0" in plane.members
    # tenant assigned by the plane is honoured for later JWTs
    assert client.identity.tenant_id == resp["tenantId"]


def test_full_round_submits_dp_update_and_pulls_new_model():
    plane = FakePlane(min_updates=1, sigma=0.0)  # sigma=0 -> deterministic check
    client = _client(plane)
    X, y = make_bank_data(2000, 0.05, seed=3)

    v_before = plane.global_version
    result = client.run_round(X, y)

    assert result.submitted is True
    assert plane.enroll_count == 1          # enrolled exactly once
    assert plane.update_count == 1
    assert result.update_norm > 0           # a real, non-trivial update was sent
    # plane aggregated -> new promoted version, pulled back by the client
    assert plane.global_version == v_before + 1
    assert result.global_version_after == plane.global_version
    # the global model actually moved
    assert np.linalg.norm(plane.global_w) > 0


def test_jwt_is_verified_against_registered_key():
    plane = FakePlane(min_updates=1)
    client = _client(plane)
    client.enroll()
    # A token from the WRONG key must be rejected by the plane's verification.
    other = NodeIdentity.generate("node0", client.identity.tenant_id)
    bad_token = other.mint_jwt()
    import pytest
    with pytest.raises(Exception):
        plane.submit_update(plane.round, {"memberId": "node0", "update": [0.0] * plane.dim}, bad_token)


def test_two_rounds_advance_version():
    plane = FakePlane(min_updates=1, sigma=0.0)
    client = _client(plane)
    X, y = make_bank_data(1500, 0.05, seed=5)
    r1 = client.run_round(X, y)
    r2 = client.run_round(X, y)
    assert r2.round == r1.round + 1
    assert plane.global_version == 2
    assert plane.enroll_count == 1          # still only enrolled once
