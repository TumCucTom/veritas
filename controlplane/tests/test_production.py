"""Production-soundness coverage: principled robust aggregation, real DP budget
accounting, honest siloRecall flow-through, SQLite persistence round-trip, the
durable signing identity, and Merkle consistency proofs."""
from __future__ import annotations

import numpy as np
import pytest

from controlplane import crypto
from controlplane.state import DIM, ControlPlane
from controlplane.transparency import (
    verify_consistency_proof,
    verify_inclusion_proof,
)
from veritas_core.dp_accountant import RDPAccountant


def _enroll4(plane):
    members = {}
    for mid in ["n0", "n1", "n2", "evil"]:
        priv, pub = crypto.generate_keypair()
        plane.enroll(mid, mid, pub)
        plane.approve(mid)
        members[mid] = priv
    return members


# --------------------------------------------------------------------------- #
# 1. Robust aggregation rejects a poisoned member BY NAME.
# --------------------------------------------------------------------------- #
def test_robust_aggregation_rejects_poisoner_by_name(plane):
    plane.min_updates = 4
    _enroll4(plane)
    base = np.random.default_rng(1).normal(0, 0.02, size=DIM)
    for i, mid in enumerate(["n0", "n1", "n2"]):
        jitter = np.random.default_rng(10 + i).normal(0, 0.004, size=DIM)
        plane.submit_update(mid, 1, list(base + jitter), 1000, {"recall": 0.83})
    plane.submit_update("evil", 1, list(-base * 50.0), 1000, {"recall": 0.0})

    result = plane.advance_round()
    assert result is not None
    assert result.rejected == ["evil"]
    assert set(result.contributors) == {"n0", "n1", "n2"}


def test_robust_aggregation_keeps_all_honest(plane):
    """An all-honest round must NOT falsely accuse anyone (Krum drops one by
    construction; the outlier gate keeps everyone)."""
    plane.min_updates = 4
    _enroll4(plane)
    for i, mid in enumerate(["n0", "n1", "n2", "evil"]):
        d = np.random.default_rng(i).normal(0, 0.05, size=DIM)
        plane.submit_update(mid, 1, list(d), 1000, {"recall": 0.8})
    result = plane.advance_round()
    assert result.rejected == []
    assert len(result.contributors) == 4


# --------------------------------------------------------------------------- #
# 2. Real DP budget: calibrated sigma is within target eps; spent eps grows.
# --------------------------------------------------------------------------- #
def test_dp_calibration_meets_target_epsilon(plane):
    acct = RDPAccountant()
    for _ in range(plane.dp_rounds):
        acct.step(plane.dp_sigma)
    eps = acct.get_epsilon(plane.dp_delta)
    assert eps <= plane.dp_target_epsilon + 1e-6
    assert plane.dp_sigma > 0


def test_spent_epsilon_increases_across_rounds(plane):
    plane.min_updates = 3
    members = ["n0", "n1", "n2"]
    for mid in members:
        priv, pub = crypto.generate_keypair()
        plane.enroll(mid, mid, pub)
        plane.approve(mid)

    assert plane.privacy_state()["spentEpsilon"] == 0.0

    spent = []
    for r in range(1, 4):
        for i, mid in enumerate(members):
            d = np.random.default_rng(r * 10 + i).normal(0, 0.05, size=DIM)
            plane.submit_update(mid, r, list(d), 1000, {"recall": 0.8})
        plane.maybe_aggregate(r)
        spent.append(plane.privacy_state()["spentEpsilon"])

    assert spent[0] < spent[1] < spent[2]
    p = plane.privacy_state()
    assert p["roundsSpent"] == 3
    assert p["noiseMultiplier"] == plane.dp_sigma
    assert p["budgetRemaining"] == round(
        p["targetEpsilon"] - p["spentEpsilon"], 6)


def test_dpparams_advertise_calibrated_sigma(plane):
    info = plane.current_round_info()
    assert info["dpParams"]["sigma"] == plane.dp_sigma
    assert info["dpParams"]["epsilon"] == plane.dp_target_epsilon
    assert info["dpParams"]["delta"] == plane.dp_delta
    assert info["dpParams"]["sigma"] != 0.05  # not the old magic value


def test_privacy_endpoint(client, make_member, honest_delta):
    body = client.get("/v1/privacy").json()
    assert set(body) == {"targetEpsilon", "delta", "spentEpsilon",
                         "noiseMultiplier", "roundsSpent", "budgetRemaining"}
    assert body["roundsSpent"] == 0
    assert body["noiseMultiplier"] > 0


# --------------------------------------------------------------------------- #
# 3. Honest metrics: reported siloRecall flows into tenant_state.
# --------------------------------------------------------------------------- #
def test_reported_silo_recall_flows_to_tenant_state(plane):
    plane.min_updates = 3
    members = ["n0", "n1", "n2"]
    for mid in members:
        priv, pub = crypto.generate_keypair()
        plane.enroll(mid, mid, pub)
        plane.approve(mid)
    for i, mid in enumerate(members):
        d = np.random.default_rng(i).normal(0, 0.05, size=DIM)
        plane.submit_update(mid, 1, list(d), 1000,
                            {"recall": 0.9, "siloRecall": 0.42})
    res = plane.advance_round()
    assert res.silo_recall == 0.42

    tid = plane.members["n0"].tenant_id
    state = plane.tenant_state(tid)
    assert state["lift"]["siloed"]["recall"] == 0.42
    assert state["lift"]["siloed"]["source"] == "reported"


def test_silo_recall_falls_back_when_unreported(plane):
    plane.min_updates = 3
    members = ["n0", "n1", "n2"]
    for mid in members:
        priv, pub = crypto.generate_keypair()
        plane.enroll(mid, mid, pub)
        plane.approve(mid)
    for i, mid in enumerate(members):
        d = np.random.default_rng(i).normal(0, 0.05, size=DIM)
        plane.submit_update(mid, 1, list(d), 1000, {"recall": 0.9})  # no siloRecall
    plane.advance_round()
    tid = plane.members["n0"].tenant_id
    state = plane.tenant_state(tid)
    assert state["lift"]["siloed"]["source"] == "heuristic"


# --------------------------------------------------------------------------- #
# 4 & 5. Persistence round-trip (SQLite) + durable signing identity.
# --------------------------------------------------------------------------- #
def test_persistence_round_trip(tmp_path):
    db = str(tmp_path / "cp.db")
    key = str(tmp_path / "plane_key.pem")

    p1 = ControlPlane(admin_key="k", min_updates=3, db_path=db, plane_key_path=key)
    members = ["n0", "n1", "n2"]
    for mid in members:
        priv, pub = crypto.generate_keypair()
        p1.enroll(mid, mid, pub)
        p1.approve(mid)
    for i, mid in enumerate(members):
        d = np.random.default_rng(i).normal(0, 0.05, size=DIM)
        p1.submit_update(mid, 1, list(d), 1000,
                         {"recall": 0.88, "siloRecall": 0.5})
    res1 = p1.advance_round()
    root1 = p1.transparency.signed_root()
    spent1 = p1.dp_spent_epsilon
    pub_key1 = p1.plane_pub_pem
    p1.store.close()

    # Reconstruct from the SAME db + key file.
    p2 = ControlPlane(admin_key="k", min_updates=3, db_path=db, plane_key_path=key)

    # Members survived.
    assert set(p2.members) == set(members)
    assert p2.members["n0"].status == "active"
    # Model registry survived (genesis + the round's new version).
    assert res1.new_version in p2.models
    assert p2.promoted_version == p1.promoted_version
    # Round result survived, incl. honest siloRecall.
    assert p2.results[1].new_version == res1.new_version
    assert p2.results[1].silo_recall == 0.5
    # Round counter advanced past the aggregated round.
    assert p2.current_round == 2
    # DP spent epsilon recovered.
    assert p2.dp_spent_epsilon == pytest.approx(spent1)
    assert p2.privacy_state()["roundsSpent"] == 1

    # Durable signing identity: same key -> transparency log still verifiable.
    assert p2.plane_pub_pem == pub_key1
    root2 = p2.transparency.signed_root()
    assert root2["rootHash"] == root1["rootHash"]
    payload = crypto.canonical_json(
        {"size": root2["size"], "rootHash": root2["rootHash"]})
    assert crypto.verify_bytes(p2.plane_pub_pem, payload, root2["signaturePem"])
    # Inclusion proof over the recovered log verifies.
    proof = p2.transparency.inclusion_proof(0)
    assert verify_inclusion_proof(proof, root2["size"]) is True
    p2.store.close()


def test_durable_key_regenerated_only_when_absent(tmp_path):
    key = str(tmp_path / "k.pem")
    a, b = crypto.load_or_create_keypair(key)
    a2, b2 = crypto.load_or_create_keypair(key)
    assert a == a2 and b == b2  # second call loads, does not regenerate


# --------------------------------------------------------------------------- #
# 6. Merkle consistency proofs (append-only-ness).
# --------------------------------------------------------------------------- #
def test_consistency_proof_verifies(plane):
    # Generate several leaves (member enrolments) to grow the tree.
    for i in range(7):
        priv, pub = crypto.generate_keypair()
        plane.enroll(f"m{i}", f"m{i}", pub)
    size = plane.transparency.size()
    assert size == 7
    for first in range(1, size + 1):
        for second in range(first, size + 1):
            proof = plane.transparency.consistency_proof(first, second)
            assert verify_consistency_proof(proof) is True


def test_consistency_proof_detects_tampering(plane):
    for i in range(6):
        priv, pub = crypto.generate_keypair()
        plane.enroll(f"m{i}", f"m{i}", pub)
    proof = plane.transparency.consistency_proof(3, 6)
    assert verify_consistency_proof(proof) is True
    # Rewriting an early leaf breaks the recomputed old root.
    proof["leaves"][1] = "ab" * 32
    assert verify_consistency_proof(proof) is False


def test_consistency_proof_endpoint(client):
    for i in range(5):
        _, pub = crypto.generate_keypair()
        client.post("/v1/members/enroll", json={
            "memberId": f"m{i}", "displayName": f"m{i}", "publicKeyPem": pub})
    proof = client.get(
        "/v1/transparency/consistency?first=2&second=5").json()
    assert proof["first"] == 2 and proof["second"] == 5
    assert verify_consistency_proof(proof) is True
