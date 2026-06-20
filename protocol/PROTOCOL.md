# Veritas Production Wire Protocol (v1)

The seam between the three tiers. Everything is JSON over HTTP. This is the source of truth that `controlplane/`, `node/`, `edge-sdk/`, and `web/` build against. Implements the architecture in `docs/superpowers/specs/2026-06-20-veritas-production-architecture.md`.

Ports (local/dev):
- Control plane (Tier 2): `:9000`
- Bank node (Tier 1): `:8100 + i` for node i (node0=:8100, node1=:8101 …); each node also serves the existing demo contract on the same port.
- Edge SDK (Tier 0): talks to its bank node's `/edge/v1/*`.

Weight vectors are JSON arrays of floats. Model dimension = `core.veritas_core.data.FEATURE_DIM + 1` (logistic bias term), currently 11. Reuse `core/veritas_core` FL primitives everywhere — do not reimplement FedAvg/Multi-Krum/DP/model.

---

## Auth & identity

Each member (bank node) has an Ed25519 keypair. Enrolment registers the public key under a tenant. Every authenticated node→plane request carries `Authorization: Bearer <jwt>` where the JWT is signed by the member's private key (EdDSA) with claims `{sub: memberId, tid: tenantId, iat, exp}`. The control plane verifies against the registered public key. Admin endpoints require an admin API key (`X-Admin-Key` header).

---

## Tier 2 — Control Plane API (`:9000`)

### Enrolment & identity
```
POST /v1/members/enroll
  body: { memberId, displayName, publicKeyPem, attestationQuote? }
  -> 201 { memberId, tenantId, status: "pending"|"active" }
POST /v1/members/{memberId}/approve        (admin)  -> { memberId, status: "active" }
GET  /v1/members                            (admin)  -> [{ memberId, displayName, tenantId, status }]
```

### Federation rounds
```
GET  /v1/rounds/current
  -> { round:int, globalModelVersion:int, status:"open"|"aggregating"|"closed",
       dpParams:{ maxNorm:float, sigma:float }, minUpdates:int }
GET  /v1/models/{version}
  -> { version:int, dim:int, weights:[float], parentVersion:int|null, status, createdAt }
GET  /v1/models/current   -> same shape as above for the promoted global model
POST /v1/rounds/{round}/updates        (auth)
  body: { memberId, round:int, update:[float], numExamples:int,
          localMetrics:{recall:float}, attestationQuote? }
  -> 202 { accepted:true, receiptId }
GET  /v1/rounds/{round}/result
  -> { round, newVersion, contributors:[memberId], rejected:[memberId],
       globalMetrics:{recall:float}, transparencySeq:int }
```
Aggregation trigger: when `#updates >= minUpdates` OR the round is closed by `POST /v1/rounds/advance` (admin/orchestrator). The plane runs **Multi-Krum** (`core.veritas_core.aggregation.multi_krum`, n_byzantine=1) over the round's updates, rejects outliers, applies the aggregate to the parent model → new version, appends a transparency record, opens the next round.

### Model registry & governance
```
GET  /v1/models/registry -> [{ version, status:"canary"|"promoted"|"rolledback", metrics, createdAt }]
POST /v1/models/{version}/promote   (admin) -> { version, status:"promoted" }
POST /v1/models/{version}/rollback  (admin) -> { version, status:"rolledback", revertedTo:int }
```

### Transparency log (Merkle, append-only, signed)
```
GET /v1/transparency        -> [{ seq, type, round?, data, leafHash, timestamp }]
GET /v1/transparency/root   -> { size:int, rootHash:hex, signaturePem }
GET /v1/transparency/proof/{seq} -> { seq, leafHash, rootHash, auditPath:[hex], index }
```
Record `type`: `round_aggregated` (data: contributors, rejected, newVersion, globalRecall, modelHash, attestationQuotes), `model_promoted`, `model_rolledback`, `member_enrolled`. Leaf = SHA-256 of canonical-JSON(record). Root signed by the control-plane Ed25519 key.

### Tenant console (per-tenant, auth or tenant-scoped)
```
GET /v1/tenants/{tenantId}/state
  -> { round, modelVersion, customerRecordsTransmitted:0,
       node:{ status, lastSync, attestation:"verified"|"unknown" },
       lift:{ federated:{recall,victims,lostGbp}, siloed:{recall,victims,lostGbp},
              fraudPreventedGbp } }
GET /v1/tenants/{tenantId}/provenance -> [{ round, contributors:[], rejected:[], globalRecall }]
GET /v1/events?tenantId=...   (SSE)  named: round_complete, model_promoted, attack_detected, member_enrolled
```

---

## Tier 1 — Bank Node API

### Outward (the bank's systems call these — the existing contract, unchanged)
```
POST /predict { transaction:{...}, text? } -> { label, confidence, indicators[], explanation? }
GET  /state    -> existing State shape (single-bank, federated-vs-siloed counterfactual)
GET  /events   (SSE) -> round_complete, client_updated, attack_detected
GET  /health   -> { ok, memberId, modelVersion, controlPlane:"connected"|"down", attestation }
```

### Connector / data ingestion (config, not code)
A declarative `feature_map.yaml` maps a data source → the `FEATURE_DIM` feature vector. The connector runtime supports source kinds: `csv` (stands in for the warehouse), `iso20022` (parse pacs.008/pain.001 XML). It produces `(X, y)` arrays consumed by the local trainer. See `node/connectors/`.

### Federation client (node→plane, internal)
The node: enrols (once), polls `/v1/rounds/current`, pulls `/v1/models/current`, trains locally on connector data (reuse `core` train_local), computes `update = local_w - global_w`, applies DP (`core.dp.privatize`), POSTs to `/v1/rounds/{round}/updates`, then pulls the new global model. Runs a loop.

### Edge aggregation (device→bank, Tier 0 endpoints)
```
GET  /edge/v1/model -> { version:int, dim:int, weights:[float] }   (bank edge model)
POST /edge/v1/updates { deviceToken, update:[float], numExamples }  -> { accepted:true }
POST /edge/v1/score   { features:[float] } -> { label, confidence, indicators[] }  (fallback; normally on-device)
```
The node **secure-aggregates** device updates (sum with DP) into the bank edge model in-tenancy; never stores per-device updates. `deviceToken` is an ephemeral enrolment token, not a customer identifier.

---

## Tier 0 — Edge SDK (device)

A TypeScript reference SDK (`edge-sdk/`). Holds a small on-device logistic model (same dim), trains locally on synthetic device-side behavioural events, applies DP to its update, and POSTs to the bank node `/edge/v1/updates`. Pulls the edge model from `/edge/v1/model`. Public API: `Veritas.start({nodeUrl, key})`, `veritas.observePayment({...}) -> {risk, reason}`, `veritas.syncModel()`, `veritas.contributeUpdate()`. Ships with a tiny browser/node demo and tests. No customer data leaves the device — only the DP-protected update.

---

## Reuse contract (do not duplicate)
- FL math: `core/veritas_core/{model,aggregation,dp,attack,data}.py`.
- Existing types: `contract/{schema.py,types.ts}` — extend, keep backward-compatible.
- The node's single-bank engine derives from `core/veritas_core/engine.py` (split: one bank's portion + a federation client, instead of simulating N banks in-process).
