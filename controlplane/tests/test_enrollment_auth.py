"""Enrolment + EdDSA-JWT auth."""
from controlplane import crypto


def test_enroll_pending_then_approve(client, admin_headers):
    priv, pub = crypto.generate_keypair()
    r = client.post("/v1/members/enroll", json={
        "memberId": "node0", "displayName": "Node Zero", "publicKeyPem": pub})
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "pending"
    assert body["tenantId"]

    # Member list is admin-gated.
    assert client.get("/v1/members").status_code == 401
    listed = client.get("/v1/members", headers=admin_headers).json()
    assert any(m["memberId"] == "node0" for m in listed)

    r2 = client.post("/v1/members/node0/approve", headers=admin_headers)
    assert r2.status_code == 200 and r2.json()["status"] == "active"


def test_admin_key_required(client):
    assert client.post("/v1/members/node0/approve").status_code == 401
    assert client.post("/v1/rounds/advance").status_code == 401


def test_jwt_auth_required_and_verified(client, make_member, honest_delta):
    m = make_member("node0")
    # Valid JWT -> accepted.
    r = m.submit(1, honest_delta(0))
    assert r.status_code == 202, r.text
    assert r.json()["accepted"] is True

    # Missing token -> 401.
    assert client.post("/v1/rounds/1/updates", json={
        "memberId": "node0", "round": 1, "update": honest_delta(0),
        "numExamples": 1, "localMetrics": {}}).status_code == 401


def test_jwt_signed_by_wrong_key_rejected(client, make_member, honest_delta):
    m = make_member("node0")
    other_priv, _ = crypto.generate_keypair()
    forged = crypto.make_member_jwt(other_priv, "node0", m.tenant_id)
    r = client.post("/v1/rounds/1/updates",
                    headers={"Authorization": f"Bearer {forged}"},
                    json={"memberId": "node0", "round": 1,
                          "update": honest_delta(0), "numExamples": 1,
                          "localMetrics": {}})
    assert r.status_code == 401


def test_pending_member_cannot_submit(client, make_member, honest_delta):
    m = make_member("node0", approve=False)
    r = m.submit(1, honest_delta(0))
    assert r.status_code == 401
