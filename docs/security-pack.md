# Veritas — Bank-Vendor Security Pack

A pre-assembled security pack to shortcut a bank's InfoSec / third-party-risk
review. It maps the Veritas architecture to **SOC 2 Type II** trust-service
criteria and **ISO/IEC 27001:2022 Annex A** control themes, and states honestly
which controls are **implemented in the reference build** versus **production
roadmap**.

References: architecture spec §8, wire protocol
[`protocol/PROTOCOL.md`](../protocol/PROTOCOL.md), `controlplane/README.md`.

The architectural thesis the whole pack rests on: **only model deltas, signed
attestations, and control messages cross the trust boundary — never customer
data.** `customerRecordsTransmitted: 0` is verifiable, not a slogan (see
`docs/dpia-template.md`).

Status legend: **[BUILD]** implemented in the reference build · **[ROADMAP]**
production hardening, interface present, not yet hardware/cert-backed.

---

## 1. SOC 2 Type II — Trust Services Criteria

### Security / Common Criteria (CC)

| TSC | Control | How Veritas satisfies it | Status |
|---|---|---|---|
| CC6.1 | Logical access — identity | Per-member **Ed25519** keypair; enrolment registers the public key under a tenant; every node→plane call carries an **EdDSA-signed JWT** (`{sub, tid, iat, exp}`) verified against the registered key. Admin ops gated by `X-Admin-Key`. (`controlplane/crypto.py`) | **[BUILD]** |
| CC6.1 | Key management | Per-node keypairs; rotation; attestation-bound enrolment. HSM-backed control-plane keys. | rotation/keys **[BUILD]**, HSM **[ROADMAP]** |
| CC6.6 | Boundary protection | **Outbound-only** networking — the node dials home; no inbound ports. One egress rule to the control-plane endpoint is the entire approved path. | **[BUILD]** (design) |
| CC6.7 | Data-in-transit | JSON over TLS between tiers; the only payloads are model deltas + attestations + control messages. | TLS **[ROADMAP]**, payload-restriction **[BUILD]** |
| CC6.1 | Tenant isolation | Each bank's node runs in its own tenancy/enclave; the control-plane console is per-tenant (`/v1/tenants/{tenantId}/state`); **no cross-bank data path exists**. | **[BUILD]** (per-tenant API), enclave isolation **[ROADMAP]** |
| CC6.8 | Malicious-input / poisoning defence | **Multi-Krum** robust aggregation (`core.veritas_core.aggregation.multi_krum`, `n_byzantine=1`) rejects Byzantine/poisoned member updates; a compromised member cannot corrupt the global model. Rejected members fire `attack_detected` and are logged. | **[BUILD]** |
| CC7.2 | Monitoring / detection | Node→plane health telemetry (no customer data); per-tenant dashboards; alerting on stale models / failed attestations / dropped rounds; `attack_detected` SSE. | telemetry/SSE **[BUILD]**, SIEM/SOC export **[ROADMAP]** |
| CC7.3 | Integrity / tamper-evidence | Signed, append-only **Merkle transparency log**; every round's contributors, rejected members, scores, model hash, and attestation quotes are recorded. Root signed by the control-plane Ed25519 key; audit proofs via `/v1/transparency/proof/{seq}`. | **[BUILD]** (in-memory log + proofs), durable store/public anchor **[ROADMAP]** |
| CC8.1 | Change management | Signed images; model registry with **canary → promote → rollback** (`/v1/models/*`); staged fleet rollout; reproducible/seeded training. Every promotion is attested in the transparency log. | registry/canary/rollback **[BUILD]**, signed-image pipeline **[ROADMAP]** |

### Confidentiality / Privacy

| TSC | Control | How Veritas satisfies it | Status |
|---|---|---|---|
| C1.1 | Confidential data stays local | Raw data never leaves the device (Tier 0) or the bank's tenancy (Tier 1). Architecturally enforced; **TEE remote attestation proves the running image can't do otherwise**. | data-locality **[BUILD]**, hardware attestation **[ROADMAP]** (software stub `node/attestation.py` today) |
| P / C | Differential privacy | Gradient clipping + Gaussian noise on **every** update, device→bank and bank→global (`core.dp.privatize`); per-tenant ε budget surfaced in governance. | DP math **[BUILD]**, ε-budget governance UI **[ROADMAP]** |
| P | Data minimisation | Only model deltas leave; the egress surface is the entire auditable boundary. | **[BUILD]** |

---

## 2. ISO/IEC 27001:2022 Annex A — Control themes

| Annex A theme | Relevant control | How Veritas satisfies it | Status |
|---|---|---|---|
| **A.5 Organizational** | A.5.7 Threat intelligence; A.5.23 cloud security | The product *is* shared fraud-threat intelligence via FL; cloud delivery via marketplace managed app / OVA. | FL **[BUILD]**, marketplace packaging **[ROADMAP]** |
| **A.5 Organizational** | A.5.34 Privacy & PII protection | DPIA template provided (`docs/dpia-template.md`); processing designed so PII never leaves the controller's tenancy. | **[BUILD]** (template + design) |
| **A.6 People** | A.6.x supplier/operator access | Node runs in a confidential-computing enclave **hardware-isolated even from the bank's own privileged operators**; Veritas operates remotely over the outbound channel only. | design **[BUILD]**, enclave isolation **[ROADMAP]** |
| **A.7 Physical** | A.7.x / confidential computing | TEE enclave (Nitro / SEV-SNP / TDX) provides hardware isolation + remote attestation. | **[ROADMAP]** (software stub today) |
| **A.8 Technological** | A.8.2/8.3 privileged & restricted access | Ed25519/JWT auth; admin key; per-tenant scoping; read-only data role scoped to agreed tables/topics. | **[BUILD]** |
| **A.8 Technological** | A.8.5 secure authentication | EdDSA-signed JWT per request, verified against registered public key; pending/unknown member or bad signature → `401`. | **[BUILD]** |
| **A.8 Technological** | A.8.12 data leakage prevention | Outbound-only design + payload restricted to deltas/attestations + TEE attestation that the image cannot exfiltrate. | design **[BUILD]**, attestation **[ROADMAP]** |
| **A.8 Technological** | A.8.15/8.16 logging & monitoring | Signed Merkle transparency log (immutable audit trail); health telemetry; `attack_detected`. | **[BUILD]** |
| **A.8 Technological** | A.8.24 cryptography | Ed25519 signing, SHA-256 Merkle leaves, DP noise; HSM-backed plane keys planned. | crypto **[BUILD]**, HSM **[ROADMAP]** |
| **A.8 Technological** | A.8.28 secure development / model governance | SR 11-7-grade model-risk trail: version history, canary results, approvals, rollback. | **[BUILD]** |
| **A.8 Technological** | A.8.32 change management | Registry canary/promote/rollback; staged signed-image rollout. | **[BUILD]** (registry), signed pipeline **[ROADMAP]** |

---

## 3. Control narrative (the seven pillars)

1. **No raw-data egress.** Architecturally enforced; only deltas + attestations
   leave. **[BUILD]** for the payload restriction and outbound-only design;
   **[ROADMAP]** for the TEE attestation that *proves* it in hardware.
2. **Differential privacy.** Clip + Gaussian noise on every update, both
   aggregation hops. **[BUILD]**.
3. **Poisoning / malicious-member defence.** Multi-Krum rejects Byzantine
   updates; rejected members logged + `attack_detected`. **[BUILD]**.
4. **Outbound-only networking.** Node dials home; no inbound ports. **[BUILD]**
   (design).
5. **Tenant isolation.** Per-tenant console; no cross-bank data path. **[BUILD]**
   at the API; enclave-level isolation **[ROADMAP]**.
6. **Merkle transparency / audit trail.** Signed append-only log + audit proofs.
   **[BUILD]**; durable store + optional public anchor **[ROADMAP]**.
7. **Key management.** Per-node Ed25519 keypairs, rotation, attestation-bound
   enrolment. **[BUILD]**; HSM-backed plane keys **[ROADMAP]**.

---

## 4. Certifications & assurance roadmap

Pre-assembled to shortcut InfoSec review (spec §8):

- **SOC 2 Type II** — control design mapped above; Type II evidence collection on
  the production roadmap.
- **ISO/IEC 27001** — Annex A mapping above; certification on the roadmap.
- **Penetration test reports** — roadmap, supplied to design partners under NDA.
- **TEE attestation evidence** — Nitro / SEV-SNP / TDX remote-quote samples once
  the hardware backend replaces the reference software stub.
- **GDPR DPIA** — fill-in template provided now (`docs/dpia-template.md`).
- **Regulatory fit** — PSR APP-reimbursement alignment (faster mule detection →
  lower 50/50 liability); SR 11-7-grade model governance trail for FCA review.
