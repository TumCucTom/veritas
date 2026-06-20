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
    update = [0.5] * dim
    resp = client.post("/edge/v1/updates", json={"deviceToken": "ephemeral-xyz", "update": update, "numExamples": 64})
    assert resp.json()["accepted"] is True
    after = np.asarray(client.get("/edge/v1/model").json()["weights"])
    assert np.linalg.norm(after - before) > 0       # the edge model moved


def test_edge_score_returns_label():
    rt, _ = _runtime()
    client = _app(rt)
    dim = rt.cfg.dim - 1  # /edge/v1/score takes FEATURE_DIM features
    r = client.post("/edge/v1/score", json={"features": [0.0] * dim})
    assert r.status_code == 200
    assert r.json()["label"] in ("fraud", "legitimate")
