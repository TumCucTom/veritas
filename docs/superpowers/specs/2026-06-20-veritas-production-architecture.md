# Veritas — Production Architecture Spec (Whole Product)

**Date:** 2026-06-20
**Status:** Draft for review. Supersedes the deployment/production sections of [`2026-06-20-veritas-design.md`](./2026-06-20-veritas-design.md) (the hackathon design spec); inherits its problem statement, regulatory thesis, and FL/DP/Multi-Krum core.
**Scope:** The whole product in production — not just the backend. Edge SDK, in-bank node, federation control plane, console, serving/integration, onboarding, operations, compliance, commercial.

---

## 0. North-star requirement

> **A bank must be able to adopt Veritas with little to no engineering work — onboarding is configuration and approvals, not software development — while keeping the honest claim that no customer data ever leaves the customer's device or the bank's tenancy.**

Every architectural decision below is scored against two axes:

1. **Bank engineering effort** — measured in *who writes code* and *how much*. Target: the bank's own engineers write ~zero backend code; the only first-party code a bank ever ships is an optional, drop-in mobile SDK using the same pattern they already use for analytics/biometrics vendors.
2. **Sovereignty integrity** — the literal truthfulness of "data never leaves." We never trade this for convenience; the clean-room option that breaks it was rejected.

"Little to no engineering" explicitly includes minimising the **InfoSec / change-approval** cost, which is the real adoption bottleneck in banks — not lines of code. Outbound-only networking, sealed images, and remote operation exist to shrink the security review, not just the build.

---

## 1. The fraud kill-chain → the three-tier federation

APP / mule fraud is a chain of events across different vantage points. No single layer can see all of it, which is exactly why the architecture is tiered. Each tier sees what it can see, keeps its data local, and shares only model math upward.

```
  KILL-CHAIN STAGE              WHO SEES IT          VERITAS TIER           DATA STAYS
  ─────────────────────────────────────────────────────────────────────────────────────
  1. Victim manipulation        the customer's       TIER 0 — Edge SDK      on the DEVICE
     (social engineering,       own device/session   (cross-device FL)
     "scam in progress")
  ─────────────────────────────────────────────────────────────────────────────────────
  2. Payment initiation         sending bank         TIER 1 — Bank Node     in the BANK's
  3. Mule receipt / fan-out     receiving bank       (cross-silo FL,        TEE / tenancy
     (the mule signature)                            TEE-isolated)
  ─────────────────────────────────────────────────────────────────────────────────────
  4. Cross-institution          nobody, today        TIER 2 — Federation    only MODEL
     campaign propagation       (the whole point)    Control Plane          DELTAS move
                                                     (FLock-orchestrated)
```

**Hierarchical federated learning.** Aggregation happens twice:

- **Device → Bank:** each customer's on-device model contributes DP-protected updates that aggregate **inside the bank's own tenancy** into the bank's edge model. Raw device data never leaves the phone; the bank never sees per-customer gradients (secure aggregation).
- **Bank → Global:** each bank node contributes DP-protected, Multi-Krum-screened updates that aggregate at the **FLock-orchestrated control plane** into the global model. Raw bank data never leaves the bank's TEE; no bank sees another bank's update.

The global model flows back down: control plane → bank node → device. A campaign signature learned at one institution reaches every member's edge within hours, and from there onto customers' devices on the next model-sync.

---

## 2. Architecture overview

```
        TIER 0 — EDGE (customer devices)                 TIER 1 — BANK (bank's own cloud/on-prem)
   ┌───────────────────────────────────┐          ┌──────────────────────────────────────────────┐
   │  Bank's consumer mobile app        │          │  Veritas Node  (sealed image, TEE enclave)     │
   │   └─ Veritas Edge SDK              │          │   ┌──────────────────────────────────────────┐ │
   │       • on-device model (TFLite/   │  secure  │   │ Edge aggregator (device→bank, in-tenancy) │ │
   │         Core ML/ExecuTorch)        │  aggreg. │   │ Silo trainer  (cross-silo FL, local data)  │ │
   │       • scam-in-progress scoring   │─────────►│   │ Local inference API  (real-time scoring)   │ │
   │       • behavioural signals        │  (DP)    │   │ Connector runtime (read-only data adapters)│ │
   │       • DATA NEVER LEAVES DEVICE   │          │   │ Attestation agent (TEE quote + transparency)│ │
   │                                    │◄─────────│   └──────────────────────────────────────────┘ │
   │       model sync (global weights)  │          │        ▲ reads bank data (local, read-only)      │
   └───────────────────────────────────┘          │        │ ── outbound-only TLS, model deltas ──┐   │
                                                   └────────┼──────────────────────────────────────┼───┘
                                                            │                                      │
                                            bank data warehouse / ISO 20022          ▼ only model deltas + signed attestations leave
                                            payment stream (Snowflake/BigQuery/             │
                                            Databricks/Kafka)                               ▼
                                                                         TIER 2 — FEDERATION CONTROL PLANE
                                                                  ┌──────────────────────────────────────────┐
                                                                  │  FLock orchestration (permissioned, crypto-│
                                                                  │  free): round coordination, member registry│
                                                                  │  Robust aggregator (Multi-Krum / trimmed)  │
                                                                  │  Global model registry + canary/rollback   │
                                                                  │  Provenance: signed append-only transparency│
                                                                  │  log (Merkle); optional on-chain anchor     │
                                                                  │  Per-tenant Console + Admin/Governance      │
                                                                  └──────────────────────────────────────────┘
```

**The trust boundary is crossed by exactly three things, none of them customer data:** model deltas (DP-clipped, noised), signed attestations/provenance records, and control messages (round start, model version). This is the entire egress surface, and it is auditable.

---

## 3. Tier 0 — the Edge SDK (on-device, cross-device FL)

**Purpose:** detect the *victim side* of APP fraud — the part that only the customer's own device can see — without any of that data leaving the device.

### 3.1 What it detects
- **Scam-in-progress:** behavioural-biometric anomalies (typing/scroll/hesitation cadence vs the user's own baseline), out-of-pattern payee, session signals consistent with coercion (e.g., screen-share / remote-access app active, an inbound call overlapping a high-value first-time transfer), socially-engineered "safe account" flows.
- **Per-user normality:** a personalised model of *this* customer's behaviour, so the unusual stands out — something a server-side cross-customer model is structurally worse at.

### 3.2 How it works on-device
- On-device model runs on **TFLite (Android) / Core ML (iOS) / ExecuTorch (cross-platform)**; inference in milliseconds, fully offline-capable.
- **Local training** on the device's own event buffer; only DP-protected gradient updates leave, via **secure aggregation** so neither the bank nor Veritas can read an individual device's update — only the aggregate.
- **Model sync** pulls the latest global/bank model on a schedule or app launch.

### 3.3 Developer experience — the "little engineering" contract for Tier 0
This is the *only* first-party code a bank ever ships, and it is deliberately the same shape as the fraud/analytics SDKs challengers already integrate (BioCatch, Sift, Feedzai, Segment):

```kotlin
// one-line init (Android shown; iOS/React Native/Flutter parity)
Veritas.start(context, key = BuildConfig.VERITAS_KEY)        // ← all the bank writes

// auto-instrumentation covers standard payment screens; explicit hook optional:
Veritas.observePayment(payeeId, amount, isNewPayee)          // returns a risk verdict + reason
```

- **Auto-instrumentation:** the SDK hooks the standard payment/transfer screens via the app's existing navigation; for most flows the bank adds *no* per-screen code.
- **No backend work:** the SDK talks only to the bank's own Veritas Node (Tier 1) and to the model-sync endpoint — the bank wires nothing server-side for Tier 0.
- **Effort estimate:** ~1 mobile engineer, ~1 sprint, mostly QA — comparable to adding any vendor SDK. This is "little engineering," not "no engineering"; Tier 1 is the "no engineering" tier.

### 3.4 Honest limits
The device cannot see the mule/receiving side or cross-bank patterns. Tier 0 is necessary-but-not-sufficient; it raises *sending-side* protection and feeds signals to Tier 1, but the core mule-detection value is Tier 1+2. We never market Tier 0 as standalone fraud prevention.

---

## 4. Tier 1 — the Bank Node (in-tenancy, TEE-isolated, cross-silo FL)

**Purpose:** the institutional brain — mule/receiving-account detection, real-time transaction scoring with cross-bank context, and the contribution of this bank's learning to the federation. **This is the zero-bank-engineering tier.**

### 4.1 Form factor — a sealed appliance the bank deploys, Veritas operates
- Shipped as a **cloud-marketplace managed application** (AWS Marketplace / Azure Managed Application / GCP Marketplace) and as an **on-prem VM/OVA** for cloud-restricted banks.
- Runs in a **confidential-computing enclave** — AWS Nitro Enclaves, AMD SEV-SNP, or Intel TDX — so the workload is hardware-isolated even from the bank's own privileged operators, and **remote attestation** produces a cryptographic quote proving the *exact* signed image is running and cannot exfiltrate data. This is what converts "trust us" into "verify us" for InfoSec.
- **Remotely operated by Veritas** via an outbound-only control channel (the node dials home; no inbound ports). Updates, model rounds, and health all flow over that one channel. The bank does not run, patch, or babysit it.

### 4.2 Data integration — config, not code (the core of "no engineering")
The bank never builds an ETL pipeline. Three design rules:

1. **Speak the standards the bank already emits.** Native ingest of **ISO 20022** (pacs.008/pain.001 payment messages), **UK Faster Payments / Pay.UK** formats, and CASS. If the bank already produces these (they do — it's mandated rail traffic), Veritas taps an existing feed.
2. **Read from where the data already lives.** Pre-built **read-only connectors** to Snowflake, BigQuery, Databricks, Postgres, and Kafka. The bank grants **one read-only role** scoped to the needed tables/topics. No new store, no export job.
3. **Veritas Solutions Engineering does the mapping — inside the bank's tenancy.** Our field team authors the declarative schema map (bank's columns → the model's feature contract) and validates it against the bank's real data *in their enclave*, where it never leaves. The bank's engineers review and approve; they don't write it.

```yaml
# Example declarative mapping — authored by Veritas SE, approved by the bank. No bank code.
feature_map:
  amount:        $.Document.FIToFICstmrCdtTrf.CdtTrfTxInf.IntrBkSttlmAmt   # ISO 20022 path
  account_age:   warehouse.accounts.opened_at -> age_days
  velocity_1h:   derived.window(payments, key=debtor, span=1h, agg=count)
  fanout:        derived.distinct(payments, key=debtor, of=creditor, span=24h)
  is_new_payee:  warehouse.payee_history.first_seen == txn.timestamp
```

### 4.3 What the bank's engineers actually do for Tier 1
A finite, config-only checklist — completed in days, not a project:

1. Deploy the managed app from the marketplace (or host the OVA). *— infra, ~hours*
2. Grant one read-only IAM role / Kafka consumer ACL scoped to the agreed data. *— infra/security*
3. Approve one outbound egress rule to the Veritas control-plane endpoint. *— network/security*
4. Review & approve the declarative feature map and the DPIA. *— fraud + DPO*
5. Choose how scores come back (§6) — usually a connector toggle.

No application code. No model code. No pipeline code.

### 4.4 Inside the node
- **Silo trainer:** local federated training on bank data (the `train`/`evaluate`/`aggregate` primitives that already exist in `core/veritas_core` — FedAvg locally, Multi-Krum at the plane, DP on every update).
- **Edge aggregator:** secure-aggregates Tier-0 device updates into the bank's edge model, in-tenancy.
- **Local inference API:** the real-time scoring endpoint the bank's systems call (§6). Sub-50ms p99 target; runs entirely local, so scoring works even if the control plane is unreachable.
- **Attestation agent:** emits TEE quotes + per-round provenance to the transparency log.

---

## 5. Tier 2 — the Federation Control Plane (FLock-orchestrated, crypto-free)

**Purpose:** coordinate rounds, robustly aggregate across banks, govern model versions, and produce tamper-evident provenance — without any bank seeing another's data or update.

### 5.1 FLock, crypto-free
- **FLock orchestrates** the federation: round coordination, member registry, and the FL Alliance substrate, using the existing `flock-sdk` `FlockModel` (`train`/`evaluate`/`aggregate`) — Veritas is already a genuine FLock model.
- Run in a **permissioned, no-wallet / no-token mode**: no on-chain settlement, no crypto for members to hold or manage. *(Open item to confirm with FLock: the exact permissioned deployment mode of FL Alliance without token incentives.)*
- **Robust aggregation:** Multi-Krum / trimmed-mean rejects poisoned or Byzantine member updates (the malicious-member defence from the core spec), so a compromised or adversarial member can't corrupt the global model.

### 5.2 Provenance without a blockchain
- Every round's contributors, rejected members, validation scores, model hash, and TEE attestation quotes are written to a **signed, append-only transparency log** built on a Merkle tree (Certificate-Transparency / Sigstore-Rekor style). This gives tamper-evidence and public verifiability **without tokens or a chain**.
- **Optional future anchor:** the Merkle root can be periodically anchored to a public ledger if a customer or regulator ever requires it — a one-line addition, off by default. The crypto-free transparency log is the production default.

### 5.3 Model lifecycle (control-plane owned, so banks don't manage it)
- **Registry + versioning** of the global model; every promotion is attested.
- **Canary + staged rollout:** a new global model is shadow-evaluated on each bank's local eval set (inside their enclave, metrics-only out) before promotion; automatic **rollback** on regression.
- **Kill-switch:** any bank can pin/freeze its local model version independent of the federation, and globally we can halt propagation instantly.

---

## 6. Serving — getting scores back with no bank engineering

Scoring is only useful if it reaches the bank's decisioning without a rebuild. Two consumption modes, both connector-driven (challengers default to the first):

1. **Standalone (default for challengers):** the node exposes a **local real-time scoring API** (`/predict` → `{label, confidence, indicators[], explanation}`) plus a thin **per-tenant analyst console** (§7). A challenger's payment service calls one local endpoint at authorisation time and acts on the verdict (allow / hold / step-up / freeze-mule). One HTTP call, against a local endpoint, with a stable contract — not an integration project.
2. **Embedded signal layer (for incumbents / entrenched stacks):** Veritas delivers the cross-bank risk score as **one additional feature/signal into the bank's existing fraud platform** (Featurespace, NICE Actimize, Feedzai) via pre-built connectors. No new console, no workflow change — the analyst keeps their current tool and simply gains a Veritas signal. Lowest disruption.

The contract is the same one already defined in `contract/` (REST + SSE: `/predict`, `/state`, `/events`, provenance), promoted to a versioned, backward-compatible public API. The hackathon seam *is* the production seam.

**Explainability:** flagged item → plain-English rationale. Production uses **FLock sovereign inference** for the explanation layer (the production swap named in the original spec); MiniMax remains the pluggable fallback. The explanation never includes data that left the bank — it runs on the local node / sovereign endpoint.

---

## 7. The product surface (not just backend)

### 7.1 Per-tenant Console (web)
Each member sees **only their own** node and the federation's aggregate — never another bank's data. Carries forward the hackathon UI, productised and multi-tenant:
- **Network health:** their node status, model version, last sync, attestation state, `customerRecordsTransmitted: 0` proof badge.
- **Detection lift:** federated-vs-baseline (the siloed counterfactual) on *their* book — the running business case.
- **Provenance ledger:** the model bill-of-materials (contributors / rejected / recall per round) backed by the transparency log; "verify" links resolve against the Merkle proof.
- **Analyst view:** inspect a flagged account/transaction with the sovereign explanation; triage queue; feedback loop (analyst confirms/dismisses → labelled signal for the next round, kept local).
- **Edge fleet:** Tier-0 coverage (app versions, % of customers on the latest model), scam-in-progress alert volumes.

### 7.2 Admin & Governance console
- **Member identity & enrolment:** onboarding workflow, key management (per-node keypairs, rotation), TEE attestation policy, role-based access.
- **Model governance:** version history, canary results, approvals, rollback — a **SR 11-7 / model-risk-management-grade** audit trail (regulated banks require this).
- **Data governance:** the active feature map, the read-scopes granted, DPIA artefacts, residency settings — all visible and exportable for audit.
- **Federation policy:** DP budget (ε) settings, aggregation parameters, member admission/suspension.

### 7.3 Alerting & SOC integration
- Real-time **scam-in-progress** and **mule-detected** alerts to the bank's existing channels (webhook, Slack/Teams, SIEM via CEF/Syslog, email).
- **Case/SAR hooks:** push enriched cases into the bank's case-management / SAR system so confirmed mule accounts feed regulatory reporting with no manual re-keying.

### 7.4 Onboarding experience (the productised "little/no engineering" journey)
A guided, mostly-Veritas-driven flow, target **first-value in ≤ 2 weeks**:
```
  Day 0   Marketplace deploy + read-role grant + egress approve  (bank infra/security, hours)
  Day 1-3 Veritas SE authors feature map in the bank's enclave; validates on real data (Veritas)
  Day 3-5 Shadow mode: node scores live traffic, no actions taken; bank watches lift in console
  Day 5-10 (optional) Mobile team integrates Edge SDK in the consumer app  (1 eng, parallel)
  Day 10-14 Go live: enable actions / signal-feed; join federation rounds
```
The bank's total *engineering* contribution: zero backend code; one optional SDK drop-in. Everything else is approvals and configuration that Veritas drives.

---

## 8. Security, privacy, compliance (the real adoption gate)

| Concern | How Veritas addresses it |
|---|---|
| **No raw data egress** | Architecturally enforced: only model deltas + attestations leave. TEE attestation *proves* the running image can't do otherwise. `customerRecordsTransmitted: 0` is verifiable, not a slogan. |
| **Device data** | Tier 0 data never leaves the phone; only secure-aggregated, DP-protected updates leave, unreadable per-device. |
| **Differential privacy** | Gradient clipping + Gaussian noise on every update, device→bank and bank→global; per-tenant ε budget surfaced in governance. |
| **Poisoning / malicious member** | Multi-Krum / trimmed-mean robust aggregation; rejected members logged to provenance; Byzantine member can't corrupt the global model. |
| **Tamper-evidence** | Signed Merkle transparency log; optional public anchor. |
| **Tenant isolation** | Each bank's node in its own tenancy/enclave; control plane is per-tenant; no cross-bank data path exists. |
| **Data residency** | UK-region control plane; node runs in the bank's chosen region; data is pinned by construction. |
| **Regulatory fit** | GDPR DPIA templates; PSR APP-reimbursement alignment (faster mule detection = lower 50/50 liability); FCA-friendly model governance (SR 11-7-grade trail). |
| **Certifications (roadmap)** | SOC 2 Type II, ISO 27001, pen-test reports, and a standard bank-vendor security pack pre-assembled to shortcut InfoSec review. |
| **Key management** | Per-node keypairs, HSM-backed control-plane keys, rotation, attestation-bound enrolment. |

---

## 9. Operations & reliability

- **Availability:** scoring is local to the node, so a control-plane outage degrades gracefully — banks keep scoring on the last-good model; only *new federated learning* pauses. Target node scoring SLA 99.9%.
- **Observability:** node→plane health telemetry (no customer data); per-tenant dashboards; alerting on stale models / failed attestations / dropped rounds.
- **Release management:** signed images, staged rollout across the fleet, canary + auto-rollback (§5.3), reproducible/seeded training for auditability.
- **Support model:** Veritas operates Tier 1/2 remotely; banks raise only config/business issues. White-glove for design-partner challengers.

---

## 10. Commercial packaging

- **Tier-based:** *Core* (Tier 1 node + standalone scoring + console), *Edge* (adds Tier 0 SDK), *Embedded* (signal-into-existing-platform connectors). Federation membership included in all.
- **Pricing aligned to value created:** the PSR liability and £-prevented framing from the original spec — banks already budget for APP-fraud reimbursement, so the ROI is a line they understand.
- **Challenger-first GTM:** land app-first challengers (one app, fast integration, cloud-native — BYOC/TEE is trivial for them), prove network lift, then use the network effect (each new member improves every member's model) to pull in incumbents via the embedded-signal path.

---

## 11. Migration from the hackathon build

What survives directly into production (high reuse):
- **The contract** (`contract/`) → the versioned public API and the node↔plane protocol. *The seam is unchanged.*
- **The FL core** (`core/veritas_core`: data/model/aggregation/dp/attack/engine + `flock_adapter`) → the silo trainer and aggregator logic, essentially as-is.
- **The console** (`web/`) → the per-tenant Console, multi-tenant-ised.

What must be built (the real production delta):
1. **Split the in-process `Engine`** (which simulates N banks in one process) into a **deployable single-bank Node + a standalone networked control plane** — the central structural change.
2. **TEE packaging + attestation** for the node; marketplace/OVA artefacts.
3. **Connector runtime** (ISO 20022 / warehouse / Kafka) + declarative feature-map engine.
4. **Tier 0 Edge SDK** (iOS/Android/RN/Flutter) + on-device training + secure aggregation.
5. **Member identity, enrolment, key management**, and **multi-tenancy/auth** (explicitly out of scope in the hackathon, in scope now).
6. **Transparency log** + (optional) anchor; production model registry, canary, rollback.
7. **SOC2/ISO27001**, DPIA templates, the bank-vendor security pack.

---

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| TEE infra coverage varies by cloud/region | Support the major three (Nitro/SEV-SNP/TDX); fall back to BYOC-without-TEE + strong contractual + outbound-only posture where TEE is unavailable; on-prem OVA for the rest. |
| "Crypto-free FLock" mode unproven | Confirm permissioned/no-token FL Alliance mode with FLock early; the transparency-log provenance is independent of FLock, so attestation doesn't depend on resolving this. |
| On-device FL battery/perf cost on phones | Train opportunistically (charging + Wi-Fi), tiny models (TFLite/ExecuTorch), strict on-device compute budget; inference-only on low-end devices. |
| Cold-start: network value needs members | Tier 0 + Tier 1 give *single-bank* value (victim-side + local mule detection) on day one, before the network exists — adoption isn't gated on critical mass. |
| Non-IID / unfair counterfactual claims | Keep the siloed-baseline counterfactual (same code, sharing off) as the honest, in-product lift measure. |
| InfoSec review still slow despite low code | Pre-assembled security pack, attestation evidence, outbound-only design, and DPIA templates target the *review*, not just the build. |

---

## 13. Out of scope (for this revision)

- Non-UK / multi-jurisdiction expansion (design for it via residency settings, don't build it yet).
- Real-time graph/network-level fraud analytics beyond the per-account model (a future tier).
- On-chain settlement/incentives (deliberately excluded — crypto-free).
- Productised marketplace billing/metering automation (manual for design partners first).

---

## 14. Open decisions to confirm

1. **FLock permissioned/no-token mode** — exact FL Alliance deployment shape without crypto (with FLock).
2. **Primary TEE target** — Nitro vs SEV-SNP vs TDX as the first-class platform (follow first design-partner challengers' clouds).
3. **Secure-aggregation protocol** for Tier 0 (e.g., Bonawitz-style) vs lighter DP-only at the edge for v1.
4. **First two design-partner challengers** — their cloud + app stack pins concrete connector/SDK priorities.

---

## Appendix A — Hackathon alignment (FLock.io × UK Sovereign AI)

This production architecture is judged indirectly: the hackathon scores the *prototype* (already built) against five criteria and three tracks. The purpose of this appendix is to show the production direction **strengthens** the submission and to pre-empt the one place it could be misread.

### A.1 Criteria mapping

| Criterion | How this architecture lands it |
|---|---|
| Clear, relevant use case | Cross-bank APP/mule fraud — multi-£bn UK problem with live PSR (Oct-2024) reimbursement urgency. |
| Decentralised / privacy-preserving AI infra | *More* of it, not less: cross-**device** FL (data never leaves the phone) **+** cross-**silo** FL (never leaves the bank) **+** DP **+** TEE attestation, all FLock-orchestrated. On-device FL is the canonical privacy-preserving-AI pattern. |
| Strong technical implementation | Working demo today; this spec is the production depth behind it. |
| Trust, governance, security | Multi-Krum poisoning defence, TEE remote attestation, signed transparency-log provenance, SR 11-7-grade model governance. |
| **Credible path to adoption** | The core contribution of this spec: "little/no bank engineering" onboarding, challenger-first GTM, commercial tiers — turning the hardest-to-evidence criterion into the strongest. |

### A.2 Track mapping
- **Primary — Trusted Data & AI Infrastructure:** "cross-institution model training," "secure AI infrastructure for regulated industries," "privacy-preserving data sharing" — all three, deepened.
- **Secondary — AI Governance, Transparency & Trust:** transparency log + model bill-of-materials + audit trail → "decentralised model governance," "AI audit tools," "compliance and regulatory tooling."
- **Sovereign-AI thesis:** UK-owned, collectively-improving model on infrastructure the institutions own, nothing leaving any silo.

### A.3 The FLock-native vs crypto-free framing (the one thing to get right)
The brief says **"Using FLock's APIs,"** and FLock's identity is decentralised + on-chain. Two of our production choices — **permissioned, crypto-free FLock** and a **Merkle transparency log instead of on-chain attestation** — are correct for regulated banks but could be misread by FLock's judges as "designing FLock's decentralisation out." They are not. We present **two production paths, not a subtraction:**

- **FLock-native path (lead with this for FLock judges):** federation orchestrated on FLock's FL Alliance; Veritas is a genuine `flock-sdk` `FlockModel`; provenance anchored **on-chain** via FLock attestation. This is the decentralised, sovereign default we demo and pitch.
- **Enterprise permissioned path (the adoption-enabler):** the *same* FLock-orchestrated federation in a no-wallet/no-token mode, with provenance in a signed Merkle transparency log and an **optional** on-chain anchor. This exists purely to get the most conservative regulated banks over the InfoSec/compliance line.

So crypto-free is an *additional adoption path on top of* FLock-native decentralisation, never a replacement. And **TEE is defence-in-depth that complements FLock** — hardware attestation *and* federated decentralisation, not instead of. In every external artefact (pitch, deck, submission), FLock orchestration + on-chain attestation is named as the production-decentralisation path; crypto-free is introduced second, as enterprise hardening.
