"""Full federation round: honest updates -> Multi-Krum aggregate -> new model
version -> round_aggregated transparency record. Plus poisoned-update rejection
firing attack_detected and listing the member as rejected."""
import numpy as np


def test_full_honest_round_creates_new_version(client, make_member, honest_delta,
                                               admin_headers):
    nodes = [make_member(f"node{i}") for i in range(3)]
    # All three submit similar honest deltas; the 3rd triggers aggregation
    # (min_updates=3).
    for i, n in enumerate(nodes):
        r = n.submit(1, honest_delta(seed=i, scale=0.05), recall=0.8 + 0.01 * i)
        assert r.status_code == 202

    # Round 1 result exists, new version minted, next round opened.
    res = client.get("/v1/rounds/1/result")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["newVersion"] == 1
    assert set(body["contributors"]) == {"node0", "node1", "node2"}
    assert body["rejected"] == []
    assert body["transparencySeq"] >= 0

    cur = client.get("/v1/rounds/current").json()
    assert cur["round"] == 2 and cur["status"] == "open"

    # New model is registered and applied (delta added to genesis zero model).
    model = client.get("/v1/models/1").json()
    assert model["version"] == 1 and model["parentVersion"] == 0
    assert model["dim"] == len(model["weights"]) == 11
    assert np.linalg.norm(model["weights"]) > 0

    # A round_aggregated record landed in the transparency log.
    log = client.get("/v1/transparency").json()
    agg = [e for e in log if e["type"] == "round_aggregated"]
    assert len(agg) == 1
    assert agg[0]["data"]["newVersion"] == 1
    assert agg[0]["round"] == 1


def test_admin_advance_aggregates_partial_round(client, make_member, honest_delta,
                                                admin_headers):
    # Only 2 updates (< min_updates=3): no auto-aggregate, admin forces it.
    a = make_member("node0"); b = make_member("node1")
    a.submit(1, honest_delta(0)); b.submit(1, honest_delta(1))
    assert "result" not in client.get("/v1/rounds/1/result").json() if \
        client.get("/v1/rounds/1/result").status_code == 200 else True
    assert client.get("/v1/rounds/1/result").status_code == 404

    r = client.post("/v1/rounds/advance", headers=admin_headers)
    assert r.status_code == 200 and r.json()["aggregated"] is True
    assert client.get("/v1/rounds/1/result").status_code == 200


def test_multikrum_rejects_poisoned_update_over_http(client, make_member,
                                                     honest_delta, admin_headers):
    """Higher min_updates so all four updates land in one round before
    aggregation; the poisoned update must end up in `rejected`."""
    client.app  # ensure app bound
    # Raise the threshold via the live plane so auto-aggregate waits for all 4.
    from controlplane.server import app as app_module
    app_module.plane.min_updates = 4

    honest = [make_member(f"node{i}") for i in range(3)]
    attacker = make_member("evil")
    base = np.asarray(honest_delta(seed=42, scale=0.02))
    for i, n in enumerate(honest):
        n.submit(1, base + np.asarray(honest_delta(seed=100 + i, scale=0.004)),
                 recall=0.82)
    attacker.submit(1, list(-base * 50.0), recall=0.0)  # 4th -> aggregates

    res = client.get("/v1/rounds/1/result")
    assert res.status_code == 200, res.text
    body = res.json()
    assert "evil" in body["rejected"]
    assert set(body["contributors"]) == {"node0", "node1", "node2"}


def test_poisoned_rejection_fires_and_lists_member(plane):
    """Drive the ControlPlane directly so all 4 updates are in one round, then
    assert Multi-Krum rejects the attacker (-> attack_detected event + rejected
    list)."""
    from controlplane import crypto

    events = []
    plane._publish = lambda ev, data: events.append((ev, data))  # capture

    # Enrol + approve 4 members.
    members = {}
    for mid in ["n0", "n1", "n2", "evil"]:
        priv, pub = crypto.generate_keypair()
        plane.enroll(mid, mid, pub)
        plane.approve(mid)
        members[mid] = priv

    base = np.random.default_rng(1).normal(0, 0.02, size=11)
    for i, mid in enumerate(["n0", "n1", "n2"]):
        jitter = np.random.default_rng(10 + i).normal(0, 0.004, size=11)
        plane.submit_update(mid, 1, list(base + jitter), 1000, {"recall": 0.83})
    plane.submit_update("evil", 1, list(-base * 50.0), 1000, {"recall": 0.0})

    result = plane.advance_round()
    assert result is not None
    assert "evil" in result.rejected
    assert set(result.contributors) == {"n0", "n1", "n2"}

    attack_events = [d for ev, d in events if ev == "attack_detected"]
    assert any(d["memberId"] == "evil" and d["rejected"] for d in attack_events)
