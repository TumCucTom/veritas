# Veritas — Bank Deployment Runbook (Production)

How a bank brings a Veritas **Tier 1 Bank Node** into production. The design
goal is **config-not-code**: the bank's engineers write zero backend code; the
only optional first-party code is a drop-in mobile SDK. Everything else is
approvals and configuration that Veritas drives.

References: architecture spec §4.3 / §7.4, wire protocol
[`protocol/PROTOCOL.md`](../protocol/PROTOCOL.md), `node/README.md`.

---

## 1. The onboarding checklist (config + approvals only)

A finite, config-only list — completed in days, not a project (spec §4.3):

| # | Step | Owner | Effort |
|---|---|---|---|
| 1 | **Deploy the sealed node image** from the cloud marketplace (AWS / Azure / GCP managed app) or host the on-prem OVA. | Bank infra | ~hours |
| 2 | **Grant one read-only role** — a single IAM role / Kafka consumer ACL scoped to the agreed tables/topics. No new store, no export job. | Bank infra/security | ~hours |
| 3 | **Approve one outbound egress rule** to the Veritas control-plane endpoint (`:9000`). **Outbound-only** — the node dials home; no inbound ports are opened. | Bank network/security | ~hours |
| 4 | **Review & approve the declarative feature map + the DPIA.** Veritas SE authors the feature map inside the bank's enclave against real data; the bank reviews, it doesn't write it. | Fraud team + DPO | review |
| 5 | **Choose how scores come back** (§4) — usually a connector toggle: standalone `/predict` vs embedded signal. | Fraud team | config |

No application code. No model code. No pipeline code.

The node runs in a **confidential-computing enclave** (AWS Nitro Enclaves / AMD
SEV-SNP / Intel TDX). On enrolment and on every round it emits a **remote
attestation quote** proving the exact signed image is running and cannot
exfiltrate data — this is what converts "trust us" into "verify us" for
InfoSec. *(In the reference build the attestation is a software stub,
`node/attestation.py`, with an identical interface; production swaps in the
hardware quote, `Quote.kind` becoming `nitro` / `sev-snp` / `tdx`.)*

---

## 2. Day-0 → Day-14 onboarding timeline

Target **first value in ≤ 2 weeks** (spec §7.4):

```
  Day 0    Marketplace deploy + read-role grant + egress approve    (bank infra/security, hours)
  Day 1-3  Veritas SE authors feature_map.yaml in the bank's enclave;
           validates on the bank's real data — data never leaves     (Veritas)
  Day 3-5  Shadow mode: node scores live traffic, no actions taken;
           bank watches detection lift in the console                (Veritas + bank fraud)
  Day 5-10 (optional, parallel) Mobile team integrates the Edge SDK  (1 engineer)
  Day 10-14 Go live: enable actions / signal-feed; join federation rounds
```

Total **engineering** contribution from the bank: zero backend code; one
optional SDK drop-in.

### Identity & enrolment (what happens on the wire)

The node holds an **Ed25519 keypair**. On first start it enrols:

```
POST  /v1/members/enroll
  { memberId, displayName, publicKeyPem, attestationQuote? }   -> 201 { status: "pending" }
```

A Veritas admin approves it (`POST /v1/members/{memberId}/approve`,
`X-Admin-Key`) → `active`. Thereafter every node→plane request carries
`Authorization: Bearer <jwt>`, an **EdDSA-signed JWT** with claims
`{sub: memberId, tid: tenantId, iat, exp}`, verified against the registered
public key. Enrolment is recorded as a `member_enrolled` record in the Merkle
transparency log.

---

## 3. Connector / feature-map configuration

The bank **never builds an ETL pipeline**. The connector runtime
(`node/connectors/`) reads a declarative `feature_map.yaml` that maps a data
source → the model's `FEATURE_DIM` feature contract. Veritas SE authors it; the
bank approves it.

Supported source kinds (`protocol/PROTOCOL.md` §"Connector / data ingestion"):

- **`csv`** — stands in for the warehouse (Snowflake / BigQuery / Databricks /
  Postgres / Kafka in production). The bank grants one read-only role scoped to
  the agreed tables.
- **`iso20022`** — parses `pacs.008` / `pain.001` payment XML. The bank already
  emits this mandated rail traffic; Veritas taps the existing feed.

**CSV (warehouse) example** — `node/sample_data/feature_map.yaml`:

```yaml
version: 1
source:
  kind: csv
  path: transactions.csv
  label_column: is_fraud
features:
  amount:      { from: amount }
  accountAge:  { from: account_age_days }
  velocity:    { from: velocity_1h }
  fanout:      { from: fanout_24h }
  isTransfer:  { from: is_transfer,        transform: bool }
  campaignSig: { from: campaign_signature, default: 0.0 }
```

**ISO 20022 example** — `node/sample_data/feature_map_iso20022.yaml`: features
use namespace-agnostic dotted element paths resolved per `CdtTrfTxInf` block
(e.g. `amount: { iso_path: "CdtTrfTxInf/IntrBkSttlmAmt", transform: zscore }`).
Payment messages carry no fraud label, so SE supplies a constant training
`label`.

Features resolve in the canonical column order; undeclared features default to
0. The active feature map, granted read-scopes, and DPIA artefacts are all
visible and exportable in the **Data Governance** console for audit.

The node is parameterised by env vars (no code edits) — see `node/README.md`:
`VERITAS_NODE_ID`, `VERITAS_TENANT_ID`, `VERITAS_PLANE_URL`,
`VERITAS_FEATURE_MAP`, etc. A node `i` listens on `:8100 + i`.

---

## 4. How scores get back to the bank

Two consumption modes, both connector-driven (spec §6). The node serves the
existing contract unchanged.

### Mode A — Standalone (default for challengers)

The node exposes a **local real-time scoring API**. The bank's payment service
calls one local endpoint at authorisation time:

```
POST /predict
  { transaction: {...}, text? }
  -> { label, confidence, indicators[], explanation? }
```

Plus `GET /state` (federated-vs-siloed counterfactual), `GET /events` (SSE:
`round_complete`, `client_updated`, `attack_detected`), and `GET /health`
(`{ ok, memberId, modelVersion, controlPlane, attestation }`). One HTTP call
against a **local** endpoint with a stable contract — not an integration
project. Target sub-50ms p99; scoring runs entirely local, so it keeps working
even if the control plane is unreachable (`/health` reports
`controlPlane: down`).

A thin per-tenant analyst console (the `web/` surface) gives triage, the
provenance ledger, and the `customerRecordsTransmitted: 0` proof badge.

### Mode B — Embedded signal (for incumbents / entrenched stacks)

Veritas delivers the cross-bank risk score as **one additional feature/signal**
into the bank's existing fraud platform (Featurespace / NICE Actimize / Feedzai)
via pre-built connectors. No new console, no workflow change — the analyst keeps
their current tool and simply gains a Veritas signal. Lowest disruption.

The choice is usually a connector toggle (checklist step 5).

---

## 5. Going live & ongoing operation

- **Shadow → live:** the node scores in shadow mode first (no actions); the bank
  watches lift in the console before enabling actions / the signal feed.
- **Federation rounds:** once active, the node polls `/v1/rounds/current`, pulls
  `/v1/models/current`, trains locally, applies DP, and POSTs its delta to
  `/v1/rounds/{round}/updates`. The plane runs **Multi-Krum** and rejects
  outliers (a poisoned member fires `attack_detected`).
- **Kill-switch / rollback:** the bank can pin/freeze its local model version
  independent of the federation; Veritas can halt global propagation instantly.
  New global models are canary-shadow-evaluated on the bank's local eval set
  (metrics-only out) before promotion, with automatic rollback on regression.
- **Operated by Veritas:** Tier 1/2 are remotely operated over the outbound-only
  channel. The bank raises only config/business issues — it does not run, patch,
  or babysit the node.
- **Availability:** node scoring SLA target 99.9%; a control-plane outage
  degrades gracefully (keep scoring on the last-good model; only new federated
  learning pauses).
