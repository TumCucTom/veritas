from veritas_core.engine import Engine
def test_federated_beats_siloed():
    e=Engine(n_banks=6,seed=7); e.inject_campaign()
    for _ in range(6): e.step()
    s=e.state(); fed=sum(b["detection"]["federated"] for b in s["banks"])/6
    silo=sum(b["detection"]["siloed"] for b in s["banks"])/6
    assert fed>silo+0.2
def test_poison_rejected():
    # Detection (the robust aggregator actually rejecting the poisoned member)
    # and recovery (recall climbing back) are now DECOUPLED: attack_detected
    # fires the round the attacker is dropped, regardless of model health. Step
    # the full poison window so both can be observed independently — note we do
    # NOT short-circuit on the first event (that would stop stepping early).
    e=Engine(n_banks=6,seed=8); e.inject_campaign(); e.inject_attack("bank0")
    detected=False
    for _ in range(6):
        if any(x["type"]=="attack_detected" for x in e.step()):
            detected=True
    # The poisoned member must have been rejected at least once...
    assert detected
    # ...and rejected on EVERY round it was poisoned (defence is not a one-off).
    assert all(p["rejected"]==["bank0"] for p in e.provenance)
    # ...and the defence must restore federated recall despite ongoing poison.
    assert max(b["detection"]["federated"] for b in e.state()["banks"])>0.7

def test_fraud_propagated_emitted():
    # The contract declares fraud_propagated{bankId,regime} and the web
    # subscribes to it; the engine must actually emit it once per regime when a
    # campaign makes fraud spread.
    e=Engine(n_banks=6,seed=3); e.inject_campaign()
    seen={}
    for _ in range(3):
        for ev in e.step():
            if ev["type"]=="fraud_propagated":
                seen[ev["data"]["regime"]]=ev["data"]
    assert set(seen)=={"federated","siloed"}
    for d in seen.values():
        assert d["bankId"].startswith("bank")
