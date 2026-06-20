from fastapi.testclient import TestClient
from server.app import app
from veritas_core.engine import Engine

KEYS = {"round", "contributors", "rejected", "globalRecall"}


def test_provenance_records_grow_and_recover():
    e = Engine(n_banks=8, seed=42); e.inject_campaign(); e.inject_attack("bank0")
    for _ in range(6): e.step()
    prov = e.provenance
    assert len(prov) == 6
    for rec in prov:
        assert KEYS <= set(rec.keys())
        assert isinstance(rec["contributors"], list) and isinstance(rec["rejected"], list)
    # global recall climbs as the federated model learns the campaign
    assert prov[-1]["globalRecall"] > prov[0]["globalRecall"]
    # the poisoned member is dropped in at least one early round
    assert any("bank0" in rec["rejected"] for rec in prov)
    # a contributing round never lists the rejected member among contributors
    for rec in prov:
        assert not (set(rec["contributors"]) & set(rec["rejected"]))


def test_provenance_endpoint_and_reset():
    c = TestClient(app)
    c.post("/sim/reset")
    assert c.get("/provenance").json() == []  # fresh engine clears provenance
    c.post("/campaign/inject", json={"typology": "safe-account-mule"})
    c.post("/attack/inject", json={"memberId": "bank0"})
    for _ in range(5): c.post("/round/step")
    prov = c.get("/provenance").json()
    assert isinstance(prov, list) and len(prov) == 5
    assert all(KEYS <= set(r.keys()) for r in prov)
    assert any("bank0" in r["rejected"] for r in prov)
