from veritas_core.engine import Engine
def test_federated_beats_siloed():
    e=Engine(n_banks=6,seed=7); e.inject_campaign()
    for _ in range(6): e.step()
    s=e.state(); fed=sum(b["detection"]["federated"] for b in s["banks"])/6
    silo=sum(b["detection"]["siloed"] for b in s["banks"])/6
    assert fed>silo+0.2
def test_poison_rejected():
    e=Engine(n_banks=6,seed=8); e.inject_campaign(); e.inject_attack("bank0")
    det=any(any(x["type"]=="attack_detected" for x in e.step()) for _ in range(4))
    assert det and max(b["detection"]["federated"] for b in e.state()["banks"])>0.7
