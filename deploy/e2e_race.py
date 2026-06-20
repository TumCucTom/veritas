"""Distributed-race e2e: drive the demo through the REAL control plane + node
processes (not the in-process engine) and assert the race diverges and the
poison beat fires.

Assumes the stack is already up (deploy/run_local.sh up 5) with nodes running
their federation loops (VERITAS_AUTOSTART_FEDERATION=1) and a blind node set.
Uses only the legacy demo contract the web consumes, served by the plane.
"""
import json
import sys
import time
import urllib.request

PLANE = "http://localhost:9000"
ADMIN = {"X-Admin-Key": "dev-admin-key"}


def _req(method, path, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(PLANE + path, data=data, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def mean_det(state, regime):
    banks = state.get("banks", [])
    if not banks:
        return 0.0
    return sum(b["detection"][regime] for b in banks) / len(banks)


def main():
    results = []

    # Approve every enrolled member so they can submit.
    members = _req("GET", "/v1/members", headers=ADMIN)
    for m in members:
        if m["status"] != "active":
            _req("POST", f"/v1/members/{m['memberId']}/approve", body={}, headers=ADMIN)
    print(f"approved {len(members)} members: {[m['memberId'] for m in members]}")

    # Baseline: banks appear with real names from enroll metadata.
    state = _req("GET", "/state")
    names = [b["name"] for b in state["banks"]]
    ok_banks = len(state["banks"]) >= 4 and any(n in names for n in ("Barclays", "NatWest", "Lloyds"))
    results.append(("a) plane /state aggregates real nodes as named banks",
                    ok_banks, f"{len(state['banks'])} banks: {names}"))

    # Inject the campaign via the plane; nodes poll, react, train, submit.
    _req("POST", "/campaign/inject", body={"typology": "safe-account-mule"})
    print("campaign injected; waiting for federation rounds to diverge ...")
    diverged = False
    detail_c = ""
    for _ in range(40):
        time.sleep(2)
        state = _req("GET", "/state")
        fed, silo = mean_det(state, "federated"), mean_det(state, "siloed")
        detail_c = f"round {state['round']} fed={fed:.3f} silo={silo:.3f} campaignActive={state['campaignActive']}"
        if state["campaignActive"] and fed - silo > 0.12:
            diverged = True
            break
    results.append(("b) campaign -> federated detection beats siloed (real nodes)",
                    diverged, detail_c))

    # Counters accrue from real reported metrics.
    c = state["counters"]
    ok_counters = c["federated"]["fraudPreventedGbp"] > 0 and c["siloed"]["victims"] >= c["federated"]["victims"]
    results.append(("c) aggregated counters accrue (fraud prevented > 0)",
                    ok_counters,
                    f"prevented £{c['federated']['fraudPreventedGbp']} fedVictims={c['federated']['victims']} siloVictims={c['siloed']['victims']}"))

    # /predict off the plane's real global model.
    pred = _req("POST", "/predict", body={"transaction": {}})
    results.append(("d) /predict from plane global model",
                    pred.get("label") in ("fraud", "legitimate"),
                    f"label={pred.get('label')} confidence={pred.get('confidence')}"))

    # Inject a malicious member; that node poisons -> Multi-Krum rejects it.
    target = next((b["id"] for b in state["banks"] if b["id"] != "node3"), state["banks"][0]["id"])
    _req("POST", "/attack/inject", body={"memberId": target})
    print(f"attack injected on {target}; waiting for rejection ...")
    rejected = False
    detail_e = ""
    for _ in range(40):
        time.sleep(2)
        state = _req("GET", "/state")
        bank = next((b for b in state["banks"] if b["id"] == target), None)
        detail_e = f"attackActive={state['attackActive']} {target}.poisoned={bank and bank['poisoned']}"
        if state["attackActive"] and bank and bank["poisoned"]:
            rejected = True
            break
    results.append(("e) malicious member poisoned update rejected (attack_detected)",
                    rejected, detail_e))

    print("\n" + "=" * 60 + "\nDISTRIBUTED RACE E2E\n" + "=" * 60)
    allok = True
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}\n        {detail}")
        allok = allok and ok
    print("=" * 60)
    print("RESULT:", "ALL PASS" if allok else "FAILURES PRESENT")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
