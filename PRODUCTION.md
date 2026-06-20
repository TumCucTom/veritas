# Veritas — Production Overview

**Status:** Reference production build on branch `veritas-production`.
**Architecture:** [`docs/superpowers/specs/2026-06-20-veritas-production-architecture.md`](docs/superpowers/specs/2026-06-20-veritas-production-architecture.md).
**Wire protocol (source of truth):** [`protocol/PROTOCOL.md`](protocol/PROTOCOL.md).

Veritas is a **federated fraud-intelligence network for UK banks**: members
collaboratively train a shared APP/mule-fraud detection model **without any
customer data ever leaving the customer's device or the bank's tenancy**. A
fraud campaign learned at one institution reaches every member within hours —
the network effect, with the data never pooled.

The north-star requirement is **adoption with little-to-no bank engineering**:
onboarding is configuration and approvals, not software development, while the
literal truth of "no customer data leaves" is preserved by construction. Only
model deltas, signed attestations, and control messages cross the trust
boundary — the entire egress surface, and it is auditable.

---

## The three-tier architecture

Fraud is a kill-chain seen from different vantage points. Each tier keeps its
data local and shares only model math upward. Aggregation happens twice
(hierarchical FL): **device → bank** (inside the bank's tenancy) and **bank →
global** (at the control plane).

| Tier | Component | Runs where | Sees | Sends upward |
|---|---|---|---|---|
| **Tier 0** | Edge SDK | the customer's device | victim-side / scam-in-progress signals | DP-protected, secure-aggregated model updates only |
| **Tier 1** | Bank Node | the bank's own cloud / on-prem (TEE enclave) | the bank's payment/account data (read-only) | DP-clipped model deltas + signed attestations only |
| **Tier 2** | Control Plane | UK-region Veritas infrastructure | nothing but model deltas + control messages | global model back down; provenance to the transparency log |

```
  TIER 0 (device)        TIER 1 (bank tenancy)            TIER 2 (control plane :9000)
  Edge SDK  ── /edge/v1/* ──►  Bank Node :8100+i  ── outbound-only TLS ──►  FLock-orchestrated
  on-device model          local trainer + DP                              Multi-Krum aggregator
  DATA STAYS ON DEVICE     /predict /state /events /health                 model registry + canary
                           DATA STAYS IN TENANCY                           Merkle transparency log
```

The trust boundary is crossed by exactly three things, **none of them customer
data**: model deltas (DP-clipped, noised), signed attestations/provenance
records, and control messages (round start, model version).

---

## Directory map

| Directory | Tier | What it is |
|---|---|---|
| `edge-sdk/` | Tier 0 | TypeScript reference Edge SDK — on-device logistic model, local training, DP, secure-agg client. `Veritas.start({nodeUrl, key})`, `observePayment(...)`. |
| `node/` | Tier 1 | Python/FastAPI Bank Node (`:8100 + i`). Local FL trainer, federation client, connector runtime (`node/connectors/`), `feature_map.yaml`, attestation agent, edge endpoints. |
| `controlplane/` | Tier 2 | Python/FastAPI Control Plane (`:9000`). Enrolment (Ed25519/JWT), rounds, Multi-Krum aggregation, model registry/canary/rollback, Merkle transparency log, tenant console API + SSE. |
| `web/` | UI | Per-tenant console (multi-tenant), points at control plane + node, provenance verify vs transparency root, edge-fleet panel. |
| `protocol/` | — | `PROTOCOL.md` — the wire seam between all tiers. The source of truth. |
| `core/` | — | `veritas_core` FL primitives (model / aggregation / dp / attack / data). Reused everywhere; never reimplemented. |
| `contract/` | — | Shared API types (pydantic + TS). The hackathon seam, promoted to the public API. |
| `deploy/` | — | Integration engineer's territory: `deploy/run_local.sh`, `deploy/docker-compose.yml`. |

---

## Real vs documented stub

We are honest about what runs in this reference build versus what is a
documented abstraction with a clear interface and a software fallback.

| Capability | Status |
|---|---|
| Networked federated learning over HTTP (plane + ≥3 independent node processes) | **Real** |
| Robust aggregation — **Multi-Krum** (`core.veritas_core.aggregation.multi_krum`, `n_byzantine=1`), poisoned member rejected | **Real** |
| Differential privacy on every update (gradient clip + Gaussian noise, `core.dp.privatize`) | **Real** |
| Identity & auth — per-member **Ed25519** keypairs, **EdDSA-signed JWT** (`{sub, tid, iat, exp}`), `X-Admin-Key` admin | **Real** |
| **Merkle** signed, append-only transparency log + audit proofs (`/v1/transparency/*`) | **Real** |
| Model registry + canary + promote/rollback | **Real** |
| Connector runtime — `csv` (warehouse stand-in) + `iso20022` (pacs.008/pain.001) + declarative `feature_map.yaml` | **Real** |
| Edge SDK device→bank secure aggregation (DP update moves the bank edge model) | **Real** |
| Per-tenant web console reading tenant state / provenance | **Real** |
| **TEE hardware attestation** (Nitro / SEV-SNP / TDX remote quote) | **Documented stub** — `node/attestation.py` ships a software stub (SHA-256 measurement + Ed25519 signature); identical interface, production swaps the implementation. `Quote.kind` is `"software-stub"` here, `"nitro"`/`"sev-snp"`/`"tdx"` in production. |
| **FLock FL Alliance** live network / on-chain attestation | **Documented abstraction** — federation is FLock-orchestrated via the `FlockModel` `train`/`evaluate`/`aggregate` primitives; this build runs the permissioned, crypto-free path (no wallet, no token, Merkle log instead of on-chain anchor). |
| **Native mobile** (TFLite / Core ML / ExecuTorch on-device) | **Documented abstraction** — Tier 0 is a TypeScript reference SDK demonstrating the on-device training + DP + secure-agg pattern; native runtimes are the production swap. |
| **Cloud marketplace** managed-app / OVA packaging | **Documented roadmap** — the node runs as a normal process here; marketplace/OVA artefacts are the production delivery vehicle. |

We never claim TEE hardware or the FLock mainnet are running in this build.

---

## How to run it

Deployment is owned by the integration engineer under `deploy/`. To bring the
federation online locally, see `deploy/run_local.sh` and
`deploy/docker-compose.yml` — these stand up the control plane (`:9000`) plus N
bank nodes (`:8100 + i`) and the edge demo, and run a real multi-node
federation end-to-end (rounds advance, the transparency log grows, a poisoned
member is rejected).

For component-level runs see `controlplane/README.md` and `node/README.md`.

Further reading:
- [`docs/deployment-runbook-production.md`](docs/deployment-runbook-production.md) — how a bank deploys a node.
- [`docs/security-pack.md`](docs/security-pack.md) — SOC 2 / ISO 27001 control map for InfoSec review.
- [`docs/dpia-template.md`](docs/dpia-template.md) — GDPR DPIA template for the bank's DPO.
