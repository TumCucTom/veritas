from veritas_core.engine import Engine
def test_counters_prevention():
    e=Engine(n_banks=6,seed=9); e.inject_campaign()
    for _ in range(6): e.step()
    c=e.state()["counters"]
    assert c["federated"]["victims"]<=c["siloed"]["victims"]
    assert c["federated"]["fraudPreventedGbp"]>=0 and e.state()["customerRecordsTransmitted"]==0
