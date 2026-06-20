"""SSE event stream: named events fire on federation activity.

We subscribe a queue directly to the plane's publisher (the same mechanism the
SSE endpoint drains) so the test is deterministic and never blocks on a live
HTTP stream.
"""
import asyncio

import numpy as np


def test_named_events_fire(plane):
    from controlplane import crypto

    q: asyncio.Queue = asyncio.Queue()
    plane.subscribe(q)

    # Enrol + approve -> member_enrolled.
    members = {}
    for mid in ["n0", "n1", "n2"]:
        priv, pub = crypto.generate_keypair()
        plane.enroll(mid, mid, pub)
        plane.approve(mid)
        members[mid] = priv

    base = np.random.default_rng(3).normal(0, 0.05, size=11)
    for i, mid in enumerate(["n0", "n1", "n2"]):
        jitter = np.random.default_rng(30 + i).normal(0, 0.01, size=11)
        plane.submit_update(mid, 1, list(base + jitter), 1000, {"recall": 0.8})
    plane.maybe_aggregate(1)            # -> round_complete
    plane.promote(1)                    # -> model_promoted

    drained = []
    while not q.empty():
        drained.append(q.get_nowait()["event"])

    assert "member_enrolled" in drained
    assert "round_complete" in drained
    assert "model_promoted" in drained


def test_attack_detected_event_fires(plane):
    from controlplane import crypto

    q: asyncio.Queue = asyncio.Queue()
    plane.subscribe(q)
    for mid in ["n0", "n1", "n2", "evil"]:
        _, pub = crypto.generate_keypair()
        plane.enroll(mid, mid, pub)
        plane.approve(mid)

    base = np.random.default_rng(1).normal(0, 0.02, size=11)
    for i, mid in enumerate(["n0", "n1", "n2"]):
        plane.submit_update(mid, 1, list(base + np.random.default_rng(10 + i)
                            .normal(0, 0.004, size=11)), 1000, {"recall": 0.83})
    plane.submit_update("evil", 1, list(-base * 50.0), 1000, {"recall": 0.0})
    plane.advance_round()

    events = []
    while not q.empty():
        events.append(q.get_nowait())
    attack = [e["data"] for e in events if e["event"] == "attack_detected"]
    assert any(d["memberId"] == "evil" for d in attack)
