"""Federation round against the FAKE in-memory plane.

enrol -> train -> submit DP update (auth'd with EdDSA JWT) -> receive new model.
No live control plane, no network.
"""
import numpy as np
import pytest

from veritas_core.live_ensemble import make_live_data

from node.attestation import SoftwareAttestor
from node.federation import FederationClient
from node.identity import NodeIdentity
from node.live_ensemble import LIVE_ENSEMBLE_DIM, block_slice

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
    data, y = make_live_data(2000, 0.05, seed=3)

    v_before = plane.global_version
    result = client.run_round(data, y)

    assert result.submitted is True
    assert plane.enroll_count == 1          # enrolled exactly once
    assert plane.update_count == 1
    assert result.update_norm > 0           # a real, non-trivial update was sent
    # plane aggregated -> new promoted version, pulled back by the client
    assert plane.global_version == v_before + 1
    assert result.global_version_after == plane.global_version
    # the global model actually moved
    assert np.linalg.norm(plane.global_w) > 0


def test_federation_wire_is_live_ensemble_dimension():
    """The submitted update is the LIVE-ENSEMBLE vector, not the flat dim-11
    logistic, and the dimension-agnostic plane aggregates it cleanly."""
    plane = FakePlane(min_updates=1, sigma=0.0)
    assert plane.dim == LIVE_ENSEMBLE_DIM
    client = _client(plane)
    data, y = make_live_data(1500, 0.06, seed=7)
    result = client.run_round(data, y)
    assert result.submitted is True
    update = np.asarray(plane.last_update_body["update"], dtype=np.float64)
    assert update.shape[0] == LIVE_ENSEMBLE_DIM
    # the logistic block (linear slice) actually moved, so federation is live
    assert np.linalg.norm(update[block_slice("logistic")]) > 0
    assert plane.global_w.shape[0] == LIVE_ENSEMBLE_DIM


def test_plane_rejects_wrong_dimension_update():
    """A wrong-length update is rejected by the dimension-agnostic plane."""
    plane = FakePlane(min_updates=1)
    client = _client(plane)
    client.enroll()
    quote = client.attestor.attest().to_wire()
    bad = {"memberId": "node0", "round": plane.round,
           "update": [0.0] * (LIVE_ENSEMBLE_DIM - 1), "attestationQuote": quote}
    with pytest.raises(ValueError):
        plane.submit_update(plane.round, bad, client.identity.mint_jwt())


def test_plane_rejects_non_finite_update():
    plane = FakePlane(min_updates=1)
    client = _client(plane)
    client.enroll()
    quote = client.attestor.attest().to_wire()
    vec = [0.0] * LIVE_ENSEMBLE_DIM
    vec[0] = float("nan")
    bad = {"memberId": "node0", "round": plane.round, "update": vec,
           "attestationQuote": quote}
    with pytest.raises(ValueError):
        plane.submit_update(plane.round, bad, client.identity.mint_jwt())


def test_plane_rejects_forged_attestation_at_enrolment():
    """A self-asserted measurement (not matching its document) is rejected."""
    plane = FakePlane(min_updates=1)
    ident = NodeIdentity.generate("node0", "tenant0")
    attestor = SoftwareAttestor(ident, image_id="img", config_digest="cfg")
    client = FederationClient(ident, plane, attestor, rng=np.random.default_rng(1))
    # Monkeypatch the attestor to emit a forged quote (measurement tampered).
    real = attestor.attest

    def forged():
        q = real()
        q.measurement = "deadbeef"  # no longer hashes from the document
        return q

    attestor.attest = forged  # type: ignore[method-assign]
    with pytest.raises(ValueError):
        client.enroll()


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
    data, y = make_live_data(1500, 0.05, seed=5)
    r1 = client.run_round(data, y)
    r2 = client.run_round(data, y)
    assert r2.round == r1.round + 1
    assert plane.global_version == 2
    assert plane.enroll_count == 1          # still only enrolled once


def test_node_consumes_plane_calibrated_sigma():
    """The node uses dpParams.sigma from the plane, not the hardcoded fallback."""
    from node.config import SIGMA
    # Plane advertises a calibrated sigma distinct from the node's fallback.
    calibrated = SIGMA + 0.37
    plane = FakePlane(min_updates=1, sigma=calibrated)
    rnd = plane.get_current_round()
    assert rnd["dpParams"]["sigma"] == calibrated
    # The client reads dpParams.sigma and feeds it to privatize(...): with a
    # larger sigma the DP noise (hence update norm) is meaningfully larger than
    # under the hardcoded fallback for the same training delta.
    data, y = make_live_data(1500, 0.05, seed=11)
    big = _client(plane).run_round(data, y)
    plane_lo = FakePlane(min_updates=1, sigma=0.0)  # no noise baseline
    small = _client(plane_lo).run_round(data, y)
    assert big.update_norm > small.update_norm


def test_localmetrics_includes_measured_silo_recall():
    """federate_once submits BOTH recall and a measured siloRecall."""
    import numpy as np

    from node.config import NodeConfig
    from node.runtime import NodeRuntime

    plane = FakePlane(min_updates=1, sigma=0.0)
    cfg = NodeConfig(node_id="node0", node_index=0, seed=0, autostart_federation=False)
    rt = NodeRuntime(cfg, transport=plane)
    rt.federate_once()

    body = plane.last_update_body
    assert body is not None
    lm = body["localMetrics"]
    assert "recall" in lm and "siloRecall" in lm
    # siloRecall is the engine's measured siloed-baseline recall (not hardcoded).
    measured = rt.engine.silo_recall()
    assert abs(lm["siloRecall"] - round(float(measured), 4)) < 1e-9
    assert 0.0 <= lm["siloRecall"] <= 1.0
