from fastapi.testclient import TestClient
from server.app import app
def test_flow():
    c=TestClient(app); c.post("/sim/reset")
    assert c.get("/state").json()["round"]==0
    c.post("/campaign/inject",json={"typology":"safe-account-mule"})
    s=c.post("/round/step").json(); assert s["round"]==1 and "federated" in s["counters"]
def test_predict():
    r=TestClient(app).post("/predict",json={"transaction":{}}).json()
    assert r["label"] in ("fraud","legitimate") and "indicators" in r
