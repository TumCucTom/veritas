"""Merkle transparency log: signed root + inclusion-proof verification."""
from controlplane import crypto
from controlplane.transparency import verify_inclusion_proof


def _enroll_n(client, admin_headers, n):
    from controlplane import crypto as c
    for i in range(n):
        _, pub = c.generate_keypair()
        client.post("/v1/members/enroll", json={
            "memberId": f"m{i}", "displayName": f"m{i}", "publicKeyPem": pub})


def test_root_is_signed_by_plane_key(client, admin_headers):
    from controlplane.server import app as app_module
    _enroll_n(client, admin_headers, 3)  # -> 3 member_enrolled leaves
    root = client.get("/v1/transparency/root").json()
    assert root["size"] == 3
    assert root["rootHash"]
    payload = crypto.canonical_json({"size": root["size"],
                                     "rootHash": root["rootHash"]})
    assert crypto.verify_bytes(app_module.plane.plane_pub_pem, payload,
                               root["signaturePem"])


def test_inclusion_proof_verifies_for_every_leaf(client, admin_headers):
    _enroll_n(client, admin_headers, 5)
    root = client.get("/v1/transparency/root").json()
    log = client.get("/v1/transparency").json()
    assert len(log) == 5
    for entry in log:
        proof = client.get(f"/v1/transparency/proof/{entry['seq']}").json()
        assert proof["leafHash"] == entry["leafHash"]
        assert proof["rootHash"] == root["rootHash"]
        assert verify_inclusion_proof(proof, root["size"]) is True


def test_tampered_proof_fails(client, admin_headers):
    _enroll_n(client, admin_headers, 4)
    root = client.get("/v1/transparency/root").json()
    proof = client.get("/v1/transparency/proof/0").json()
    proof["leafHash"] = "00" * 32  # tamper
    assert verify_inclusion_proof(proof, root["size"]) is False


def test_leaf_hash_matches_canonical_json(client, admin_headers):
    _enroll_n(client, admin_headers, 1)
    entry = client.get("/v1/transparency").json()[0]
    rebuilt = {"seq": entry["seq"], "type": entry["type"],
               "data": entry["data"], "timestamp": entry["timestamp"]}
    assert crypto.leaf_hash(rebuilt) == entry["leafHash"]
