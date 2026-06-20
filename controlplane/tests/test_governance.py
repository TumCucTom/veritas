"""Model registry: promote + rollback + tenant console + provenance."""
import numpy as np


def _run_round(client, make_member, honest_delta):
    nodes = [make_member(f"node{i}") for i in range(3)]
    base = np.asarray(honest_delta(seed=7, scale=0.05))
    for i, n in enumerate(nodes):
        n.submit(1, base + np.asarray(honest_delta(seed=20 + i, scale=0.01)),
                 recall=0.8)
    return client.get("/v1/rounds/1/result").json()


def test_registry_lists_genesis_and_canary(client, make_member, honest_delta):
    _run_round(client, make_member, honest_delta)
    reg = client.get("/v1/models/registry").json()
    versions = {m["version"]: m["status"] for m in reg}
    assert versions[0] == "promoted"   # genesis
    assert versions[1] == "canary"     # freshly aggregated


def test_promote_then_rollback(client, make_member, honest_delta, admin_headers):
    _run_round(client, make_member, honest_delta)

    pr = client.post("/v1/models/1/promote", headers=admin_headers)
    assert pr.status_code == 200 and pr.json()["status"] == "promoted"
    assert client.get("/v1/models/current").json()["version"] == 1

    rb = client.post("/v1/models/1/rollback", headers=admin_headers)
    assert rb.status_code == 200
    body = rb.json()
    assert body["status"] == "rolledback" and body["revertedTo"] == 0
    assert client.get("/v1/models/current").json()["version"] == 0

    reg = {m["version"]: m["status"] for m in
           client.get("/v1/models/registry").json()}
    assert reg[1] == "rolledback" and reg[0] == "promoted"


def test_promote_rollback_require_admin(client):
    assert client.post("/v1/models/1/promote").status_code == 401
    assert client.post("/v1/models/1/rollback").status_code == 401


def test_governance_records_in_transparency(client, make_member, honest_delta,
                                            admin_headers):
    _run_round(client, make_member, honest_delta)
    client.post("/v1/models/1/promote", headers=admin_headers)
    client.post("/v1/models/1/rollback", headers=admin_headers)
    types = [e["type"] for e in client.get("/v1/transparency").json()]
    assert "model_promoted" in types and "model_rolledback" in types


def test_tenant_state_and_provenance(client, make_member, honest_delta,
                                     admin_headers):
    nodes = [make_member(f"node{i}") for i in range(3)]
    base = np.asarray(honest_delta(seed=7, scale=0.05))
    for i, n in enumerate(nodes):
        n.submit(1, base + np.asarray(honest_delta(seed=20 + i, scale=0.01)),
                 recall=0.8)
    client.post("/v1/models/1/promote", headers=admin_headers)

    tenant = nodes[0].tenant_id
    state = client.get(f"/v1/tenants/{tenant}/state").json()
    assert state["customerRecordsTransmitted"] == 0
    assert state["node"]["status"] == "active"
    assert state["node"]["attestation"] == "verified"
    assert state["lift"]["federated"]["recall"] >= state["lift"]["siloed"]["recall"]
    assert "fraudPreventedGbp" in state["lift"]

    prov = client.get(f"/v1/tenants/{tenant}/provenance").json()
    assert prov and prov[0]["round"] == 1
    assert set(prov[0]["contributors"]) == {"node0", "node1", "node2"}
