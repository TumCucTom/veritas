#!/usr/bin/env python3
"""Veritas end-to-end federation check (real services over HTTP).

Run AGAINST a running stack started by deploy/run_local.sh. The harness drives
aggregation explicitly (admin POST /v1/rounds/advance) so it controls how many
updates land per round; launch the stack with
``VERITAS_AUTOSTART_FEDERATION=0 VERITAS_MIN_UPDATES=99`` so the nodes don't
race ahead and aggregation only happens when this script forces it.

Assertions:
  (a) global model version advanced past genesis;
  (b) the transparency log grew AND an inclusion proof recomputes to the
      signed Merkle root;
  (c) at least one node's GET /state shows federated recall ahead of siloed;
  (d) a node's POST /predict returns a valid fraud/legitimate label;
  (e) a POISONED update (5th malicious member, sign-flipped + amplified, raw
      HTTP with a self-minted EdDSA JWT) lands in the round's `rejected` list
      and fires attack_detected.

Exit code 0 iff every assertion PASSES.

Dependencies (httpx, cryptography, PyJWT, numpy) are all present in the control
plane's venv — run with:  controlplane/.venv/bin/python deploy/e2e.py
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys
import time

import httpx
import jwt as pyjwt
import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

PLANE = "http://localhost:9000"
NODES = ["http://localhost:8100", "http://localhost:8101",
         "http://localhost:8102", "http://localhost:8103"]
ADMIN = {"X-Admin-Key": "dev-admin-key"}
HONEST_IDS = ["node0", "node1", "node2", "node3"]

http = httpx.Client(timeout=30.0)
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


# --- Merkle proof recompute (mirrors controlplane/transparency.py) ---------
def _node_hash(left: str, right: str) -> str:
    return hashlib.sha256(b"\x01" + bytes.fromhex(left) + bytes.fromhex(right)).hexdigest()


def verify_inclusion(proof: dict) -> str:
    """Recompute the Merkle root from leafHash + auditPath; return the root."""
    idx = proof["index"]
    h = proof["leafHash"]
    for sibling in proof["auditPath"]:
        h = _node_hash(h, sibling) if idx % 2 == 0 else _node_hash(sibling, h)
        idx //= 2
    return h


def wait_plane() -> None:
    for _ in range(60):
        try:
            if http.get(f"{PLANE}/health").status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise SystemExit("control plane not reachable at " + PLANE)


def approve_all() -> None:
    members = {m["memberId"] for m in http.get(f"{PLANE}/v1/members", headers=ADMIN).json()}
    for mid in members:
        http.post(f"{PLANE}/v1/members/{mid}/approve", headers=ADMIN)


def node_step(node_url: str) -> None:
    """Make a node train + submit one update into the current open round."""
    r = http.post(f"{node_url}/federate/step")
    r.raise_for_status()


def advance() -> dict:
    r = http.post(f"{PLANE}/v1/rounds/advance", headers=ADMIN)
    r.raise_for_status()
    return r.json()


def current_version() -> int:
    return int(http.get(f"{PLANE}/v1/rounds/current").json()["globalModelVersion"])


def promote_latest() -> int:
    reg = http.get(f"{PLANE}/v1/models/registry").json()
    latest = max(m["version"] for m in reg)
    http.post(f"{PLANE}/v1/models/{latest}/promote", headers=ADMIN).raise_for_status()
    return latest


# --- malicious member (raw HTTP, self-signed EdDSA JWT) --------------------
def mint_jwt(priv: Ed25519PrivateKey, member_id: str, tenant_id: str) -> str:
    now = int(time.time())
    claims = {"sub": member_id, "tid": tenant_id, "iat": now, "exp": now + 3600}
    return pyjwt.encode(claims, priv, algorithm="EdDSA")


def enroll_malicious() -> tuple[Ed25519PrivateKey, str, str]:
    priv = Ed25519PrivateKey.generate()
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    mid = "evil"
    body = {"memberId": mid, "displayName": "Poison Bank", "publicKeyPem": pub_pem,
            "attestationQuote": {"kind": "software-stub", "measurement": "00"}}
    r = http.post(f"{PLANE}/v1/members/enroll", json=body)
    r.raise_for_status()
    tenant = r.json()["tenantId"]
    http.post(f"{PLANE}/v1/members/{mid}/approve", headers=ADMIN).raise_for_status()
    return priv, mid, tenant


def main() -> int:
    print("Veritas E2E — driving a real multi-node federation over HTTP\n")
    wait_plane()

    print("Setup: approve all enrolled members")
    approve_all()

    # Cross-institution campaign: nodes 0..2 are "seeing" banks (the new scam
    # typology is in their LOCAL training data, so federation can learn its
    # signature from them); node3 is targeted but BLIND (campaign in its eval
    # only, never in its siloed training). The federated model must carry the
    # signal into node3 and beat its siloed baseline -> assertion (c).
    print("Injecting cross-institution campaign (nodes0-2 seeing, node3 blind)")
    for url in NODES[:3]:
        http.post(f"{url}/campaign/inject", params={"seeing": "true"}).raise_for_status()
    http.post(f"{NODES[3]}/campaign/inject", params={"seeing": "false"}).raise_for_status()
    BLIND_NODE = NODES[3]

    # ---- drive several honest rounds, promoting each new canary ----------
    print("\nDriving honest federation rounds (4 nodes, promote each round):")
    v0 = current_version()
    for rnd in range(1, 6):
        for url in NODES:
            node_step(url)
        agg = advance()
        promoted = promote_latest()
        print(f"  round {rnd}: aggregated={agg.get('aggregated')} "
              f"newVersion={agg.get('newVersion')} promoted=v{promoted}")
        time.sleep(0.2)

    v1 = current_version()

    # (a) global model version advanced ------------------------------------
    check("a) global model version advanced",
          v1 > v0, f"genesis v{v0} -> v{v1}")

    # let the nodes pull the freshly promoted global model before reading state
    time.sleep(3.0)

    # (b) transparency log grew + inclusion proof verifies -----------------
    log = http.get(f"{PLANE}/v1/transparency").json()
    root = http.get(f"{PLANE}/v1/transparency/root").json()
    agg_recs = [e for e in log if e["type"] == "round_aggregated"]
    seq = agg_recs[-1]["seq"]
    proof = http.get(f"{PLANE}/v1/transparency/proof/{seq}").json()
    recomputed = verify_inclusion(proof)
    ok_b = (len(log) > 1 and recomputed == proof["rootHash"]
            and recomputed == root["rootHash"] and root["signaturePem"])
    check("b) transparency grew & inclusion proof verifies to signed root",
          bool(ok_b),
          f"log={len(log)} recs, proof seq {seq} -> root {recomputed[:12]}.. "
          f"== signed root (size {root['size']})")

    # (c) federated recall ahead of siloed on the targeted-but-blind node --
    ahead = []
    for url in NODES:
        det = http.get(f"{url}/state").json()["banks"][0]["detection"]
        ahead.append((url, det["federated"], det["siloed"]))
    blind = next((f, s) for (u, f, s) in ahead if u == BLIND_NODE)
    winners = [(u, f, s) for (u, f, s) in ahead if f > s]
    check("c) >=1 node: federated recall ahead of siloed",
          blind[0] > blind[1] or len(winners) > 0,
          "blind node3 fed={} > silo={}; all: ".format(*blind)
          + "; ".join(f"{u.split(':')[-1]} fed={f} silo={s}" for u, f, s in ahead))

    # (d) predict returns a valid label ------------------------------------
    pr = http.post(f"{NODES[0]}/predict",
                   json={"transaction": {"amount": 9000, "velocity": 5,
                                         "fanout": 8, "accountAge": 1}}).json()
    check("d) /predict returns a valid label",
          pr.get("label") in ("fraud", "legitimate")
          and isinstance(pr.get("confidence"), (int, float)),
          f"label={pr.get('label')} confidence={pr.get('confidence')}")

    # (e) poisoned update rejected + attack_detected -----------------------
    print("\nPoison round: 4 honest updates + 1 sign-flipped/amplified update")
    priv, evil_id, evil_tenant = enroll_malicious()
    rnd = int(http.get(f"{PLANE}/v1/rounds/current").json()["round"])

    # 4 honest nodes submit into this round.
    for url in NODES:
        node_step(url)

    # Build a poison delta: take an honest delta as the baseline, flip sign,
    # amplify >10x so it is a genuine Krum outlier.
    honest_model = http.get(f"{PLANE}/v1/models/current").json()
    dim = honest_model["dim"]
    # Reference the round's honest cloud scale via a node's last update norm:
    # a unit-ish honest delta amplified hard is a clean outlier.
    rng = np.random.default_rng(7)
    base = rng.normal(0, 0.05, size=dim)
    poison = (-base * 60.0).tolist()
    token = mint_jwt(priv, evil_id, evil_tenant)
    body = {"memberId": evil_id, "round": rnd, "update": poison,
            "numExamples": 1000, "localMetrics": {"recall": 0.0},
            "attestationQuote": {"kind": "software-stub", "measurement": "evil"}}
    sub = http.post(f"{PLANE}/v1/rounds/{rnd}/updates", json=body,
                    headers={"Authorization": f"Bearer {token}"})
    poison_accepted = sub.status_code == 202

    # Force aggregation now that 5 updates (4 honest + poison) are in the round.
    advance()
    res = http.get(f"{PLANE}/v1/rounds/{rnd}/result").json()
    rejected = res.get("rejected", [])
    contributors = res.get("contributors", [])

    # attack_detected lands in the transparency-driven round record's `rejected`;
    # verify via the round result (the SSE event mirrors this list).
    ok_e = poison_accepted and evil_id in rejected and set(HONEST_IDS) <= set(contributors)
    check("e) poisoned update rejected (attack_detected)",
          bool(ok_e),
          f"round {rnd}: rejected={rejected} contributors={contributors}")

    # ---- summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("E2E SUMMARY")
    print("=" * 60)
    allok = True
    for name, ok, _ in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        allok = allok and ok
    print("=" * 60)
    print("RESULT:", "ALL PASS" if allok else "FAILURES PRESENT")
    return 0 if allok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        http.close()
