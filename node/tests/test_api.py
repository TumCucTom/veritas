"""Outward + edge API tests via FastAPI TestClient, federation injected.

We build a NodeRuntime wired to the FakePlane (no network), run a couple of
rounds so the model is trained, and assert /predict returns a valid label and
edge secure-aggregation moves the edge model.
"""
import numpy as np
from fastapi.testclient import TestClient

from node.config import NodeConfig
from node.runtime import NodeRuntime

from .fake_plane import FakePlane


def _runtime(tmp_index=0):
    cfg = NodeConfig(node_id=f"node{tmp_index}", node_index=tmp_index, seed=tmp_index,
                     autostart_federation=False)
    plane = FakePlane(min_updates=1, sigma=0.0)
    rt = NodeRuntime(cfg, transport=plane)
    return rt, plane


def _app(rt):
    # Build a minimal app bound to this runtime (avoid the module-global app).
    from fastapi import FastAPI
    from node.server import app as appmod
    # Reuse route functions by swapping the module-level runtime.
    appmod.runtime = rt
    return TestClient(appmod.app)


def test_predict_returns_valid_label():
    rt, plane = _runtime()
    # train via a couple of federation rounds
    for _ in range(3):
        rt.federate_once()
    client = _app(rt)
    r = client.post("/predict", json={"transaction": {"campaignSignature": 1.0}})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("fraud", "legitimate")
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["indicators"], list)


def test_state_has_federated_vs_siloed():
    rt, plane = _runtime()
    rt.federate_once()
    client = _app(rt)
    s = client.get("/state").json()
    det = s["banks"][0]["detection"]
    assert "federated" in det and "siloed" in det
    assert s["customerRecordsTransmitted"] == 0


def test_health_reports_member_and_attestation():
    rt, _ = _runtime()
    client = _app(rt)
    h = client.get("/health").json()
    assert h["memberId"] == "node0"
    assert "attestation" in h
    assert h["controlPlane"] in ("connected", "down")


def test_edge_secure_aggregation_moves_edge_model():
    rt, _ = _runtime()
    client = _app(rt)
    before = np.asarray(client.get("/edge/v1/model").json()["weights"])
    dim = len(before)

    # Open a cohort of two devices and deal pairwise seeds.
    opened = client.post(
        "/edge/v1/cohort/open", json={"clientIds": ["devA", "devB"]}).json()
    cohort = opened["cohortId"]

    # Each device masks its (DP) update before sending; the node only ever sees
    # the masked vectors. We reconstruct the masks here to post the masked wire
    # payloads, mirroring what the SDK does on-device.
    from veritas_core.secure_agg import mask_update
    seeds = {tuple(k.split("|")): bytes.fromhex(v) for k, v in opened["seedTable"].items()}
    uA = np.array([0.5] * dim)
    uB = np.array([0.2] * dim)
    mA = mask_update(uA, "devA", ["devB"], seeds)
    mB = mask_update(uB, "devB", ["devA"], seeds)

    for cid, mv in (("devA", mA), ("devB", mB)):
        resp = client.post("/edge/v1/updates", json={
            "deviceToken": f"ephemeral-{cid}", "update": mv.tolist(),
            "numExamples": 64, "cohortId": cohort, "clientId": cid})
        assert resp.json()["accepted"] is True

    client.post("/edge/v1/cohort/close", json={"cohortId": cohort})
    after = np.asarray(client.get("/edge/v1/model").json()["weights"])
    assert np.linalg.norm(after - before) > 0       # the edge model moved
    # The folded delta equals the secure SUM (uA+uB)/n, NOT either individual.
    expected = (uA + uB) / 2.0
    assert np.allclose(after - before, expected, atol=1e-6)


def test_edge_score_returns_label():
    rt, _ = _runtime()
    client = _app(rt)
    dim = rt.cfg.dim - 1  # /edge/v1/score takes FEATURE_DIM features
    r = client.post("/edge/v1/score", json={"features": [0.0] * dim})
    assert r.status_code == 200
    assert r.json()["label"] in ("fraud", "legitimate")


def test_edge_updates_reject_wrong_dimension():
    rt, _ = _runtime()
    client = _app(rt)
    good_dim = rt.cfg.dim  # edge update is the edge-model dimension
    r = client.post("/edge/v1/updates", json={
        "deviceToken": "d", "update": [0.0] * (good_dim - 1), "numExamples": 1})
    assert r.status_code == 422


def test_edge_validate_vector_rejects_non_finite():
    """The server-side validator rejects NaN/inf (JSON can't even carry inf, so
    we exercise the guard directly — a non-finite value would poison the sum)."""
    import numpy as np
    from fastapi import HTTPException

    from node.server.app import _validate_vector

    bad = [0.0, float("nan"), 0.0]
    try:
        _validate_vector(bad, 3, what="edge update")
        assert False, "non-finite vector was accepted"
    except HTTPException as exc:
        assert exc.status_code == 422
    # a finite vector of the right length passes
    out = _validate_vector([1.0, 2.0, 3.0], 3, what="edge update")
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_edge_score_rejects_wrong_dimension():
    rt, _ = _runtime()
    client = _app(rt)
    r = client.post("/edge/v1/score", json={"features": [0.0] * (rt.cfg.dim - 2)})
    assert r.status_code == 422
