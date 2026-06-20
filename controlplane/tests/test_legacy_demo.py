"""Legacy single-backend demo contract served by the control plane (:9000).

These tests assert the EXACT shape the web's useVeritas store consumes, backed
by aggregating the real enrolled federation members. The web points
NEXT_PUBLIC_API_BASE at the plane with zero web changes.
"""
from __future__ import annotations

import numpy as np

from controlplane.state import DIM

from controlplane import crypto


def _enroll_with_meta(client, admin_headers, member_id, display_name, customers,
                      approve=True):
    """Enrol a member with the optional displayName + customers demo metadata,
    returning a FakeMember-like signer."""
    priv, pub = crypto.generate_keypair()
    r = client.post("/v1/members/enroll", json={
        "memberId": member_id,
        "displayName": display_name,
        "publicKeyPem": pub,
        "customers": customers,
        "attestationQuote": f"quote-{member_id}",
    })
    assert r.status_code == 201, r.text
    if approve:
        r2 = client.post(f"/v1/members/{member_id}/approve", headers=admin_headers)
        assert r2.status_code == 200, r2.text
    return priv, r.json()["tenantId"]


def test_enroll_metadata_surfaces_in_state_banks(client, admin_headers):
    _enroll_with_meta(client, admin_headers, "barclays", "Barclays", 2_100_000)
    banks = client.get("/state").json()["banks"]
    assert len(banks) == 1
    b = banks[0]
    assert b["id"] == "barclays"
    assert b["name"] == "Barclays"
    assert b["customers"] == 2_100_000
    assert b["poisoned"] is False
    assert b["detection"] == {"federated": 0.0, "siloed": 0.0}


def test_enroll_defaults_display_name_to_member_id(client, admin_headers):
    # No displayName / customers -> defaults.
    priv, pub = crypto.generate_keypair()
    r = client.post("/v1/members/enroll", json={
        "memberId": "lloyds", "publicKeyPem": pub})
    assert r.status_code == 201, r.text
    client.post("/v1/members/lloyds/approve", headers=admin_headers)
    b = client.get("/banks").json()[0]
    assert b["name"] == "lloyds" and b["customers"] == 0


def test_reported_metrics_drive_banks_and_counters_accrue(client, make_member,
                                                          honest_delta):
    """After several members submit {recall, siloRecall}, banks[] show the
    reported federated/siloed detection and counters accrue: federated
    fraudPreventedGbp grows when siloed mean < federated mean."""
    nodes = [make_member(f"node{i}") for i in range(3)]
    base = np.asarray(honest_delta(seed=7, scale=0.02))
    # Federated detection HIGH, siloed LOW -> siloed loses more customers.
    for i, n in enumerate(nodes):
        jitter = np.asarray(honest_delta(seed=50 + i, scale=0.004))
        n.submit(1, base + jitter, recall=0.95, silo_recall=0.5)

    state = client.get("/state").json()
    banks = state["banks"]
    assert len(banks) == 3
    for b in banks:
        assert b["detection"]["federated"] == 0.95
        assert b["detection"]["siloed"] == 0.5

    counters = state["counters"]
    # mean fed=0.95 -> fewer victims; mean silo=0.5 -> more victims.
    assert counters["siloed"]["victims"] > counters["federated"]["victims"]
    assert counters["federated"]["fraudPreventedGbp"] > 0
    assert counters["federated"]["lostGbp"] == counters["federated"]["victims"] * 255
    assert counters["siloed"]["lostGbp"] == counters["siloed"]["victims"] * 255
    # State top-level shape the web expects.
    assert state["running"] is True
    assert state["customerRecordsTransmitted"] == 0
    assert "campaignActive" in state and "attackActive" in state


def test_campaign_and_attack_inject_set_flags(client, make_member, admin_headers):
    make_member("node0")  # need a member for attack target realism
    # Demo controls mutate plane state -> admin-gated.
    assert client.post("/campaign/inject", json={"typology": "x"}).status_code == 401
    r1 = client.post("/campaign/inject", json={"typology": "APP fraud"},
                     headers=admin_headers)
    assert r1.status_code == 200 and r1.json() == {"ok": True}
    r2 = client.post("/attack/inject", json={"memberId": "node0"},
                     headers=admin_headers)
    assert r2.status_code == 200 and r2.json() == {"ok": True}

    state = client.get("/state").json()
    assert state["campaignActive"] is True
    assert state["attackActive"] is True

    cur = client.get("/v1/rounds/current").json()
    assert cur["campaignActive"] is True
    assert cur["attackMemberId"] == "node0"
    assert "epoch" in cur


def test_round_step_advances_and_returns_state(client, make_member, honest_delta,
                                               admin_headers):
    # Raise threshold so updates accumulate without auto-aggregating.
    from controlplane.server import app as app_module
    app_module.plane.min_updates = 99
    nodes = [make_member(f"node{i}") for i in range(3)]
    base = np.asarray(honest_delta(seed=3, scale=0.02))
    for i, n in enumerate(nodes):
        n.submit(1, base + np.asarray(honest_delta(seed=80 + i, scale=0.004)),
                 recall=0.9, silo_recall=0.6)

    before = client.get("/state").json()["round"]
    assert client.post("/round/step").status_code == 401  # admin-gated
    state = client.post("/round/step", headers=admin_headers).json()
    assert state["round"] == before + 1
    assert "banks" in state and "counters" in state
    # Global model advanced + auto-promoted.
    assert app_module.plane.promoted_version >= 1


def test_sim_reset_bumps_epoch_and_clears(client, make_member, honest_delta,
                                          admin_headers):
    from controlplane.server import app as app_module
    app_module.plane.min_updates = 3
    nodes = [make_member(f"node{i}") for i in range(3)]
    base = np.asarray(honest_delta(seed=5, scale=0.02))
    for i, n in enumerate(nodes):
        n.submit(1, base + np.asarray(honest_delta(seed=90 + i, scale=0.004)),
                 recall=0.9, silo_recall=0.5)
    client.post("/campaign/inject", json={"typology": "x"}, headers=admin_headers)

    epoch_before = client.get("/v1/rounds/current").json()["epoch"]
    assert client.post("/sim/reset").status_code == 401  # admin-gated
    state = client.post("/sim/reset", headers=admin_headers).json()
    assert state["round"] == 1
    assert state["campaignActive"] is False
    assert state["attackActive"] is False
    # Counters cleared.
    assert state["counters"]["federated"]["victims"] == 0
    assert state["counters"]["siloed"]["victims"] == 0
    cur = client.get("/v1/rounds/current").json()
    assert cur["epoch"] == epoch_before + 1
    assert app_module.plane.promoted_version == 0


def test_predict_returns_valid_label(client):
    body = client.post("/predict", json={
        "transaction": {"campaignSignature": 1.0}}).json()
    assert body["label"] in {"fraud", "legitimate"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["indicators"], list) and body["indicators"]


def test_events_emit_round_complete_with_state(plane):
    """On each round aggregation the plane publishes the legacy demo events the
    web store consumes: round_complete (full State), client_updated per bank,
    and attack_detected for rejected members. We subscribe a queue directly to
    the publisher (same mechanism the SSE endpoint drains) so the test never
    blocks on a live HTTP stream."""
    import asyncio

    q: asyncio.Queue = asyncio.Queue()
    plane.subscribe(q)
    for mid in ["n0", "n1", "n2"]:
        _, pub = crypto.generate_keypair()
        plane.enroll(mid, mid, pub, customers=1000)
        plane.approve(mid)

    base = np.random.default_rng(3).normal(0, 0.02, size=DIM)
    for i, mid in enumerate(["n0", "n1", "n2"]):
        jitter = np.random.default_rng(30 + i).normal(0, 0.004, size=DIM)
        plane.submit_update(mid, 1, list(base + jitter), 1000,
                            {"recall": 0.9, "siloRecall": 0.5})
    plane.demo_step()  # aggregate + auto-promote

    events = []
    while not q.empty():
        events.append(q.get_nowait())

    round_complete = [e["data"] for e in events if e["event"] == "round_complete"]
    assert round_complete, "no round_complete event"
    state = round_complete[-1]
    assert "banks" in state and "counters" in state and "round" in state
    assert len(state["banks"]) == 3

    client_updated = [e["data"] for e in events if e["event"] == "client_updated"]
    assert {e["bankId"] for e in client_updated} == {"n0", "n1", "n2"}
    assert all("detection" in e for e in client_updated)
