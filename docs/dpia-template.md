# Data Protection Impact Assessment (DPIA) — Veritas Bank Node

**Template for a bank's Data Protection Officer.** Veritas is a federated
fraud-intelligence network: members train a shared APP/mule-fraud model
**without any customer data leaving the customer's device or the bank's
tenancy**. Only DP-protected model deltas, signed attestations, and control
messages cross the trust boundary.

Fill in the `‹…›` fields. References: architecture spec §8, wire protocol
[`protocol/PROTOCOL.md`](../protocol/PROTOCOL.md). This template is a starting
point, not legal advice — adapt to your institution's DPIA framework and ICO
guidance.

---

## 0. Document control

| Field | Value |
|---|---|
| Controller | ‹Bank legal entity› |
| Processor | Veritas ‹entity› (operates Tier 1 node + Tier 2 control plane) |
| DPO / author | ‹name, contact› |
| Version / date | ‹v / date› |
| Tier(s) in scope | ☐ Tier 0 Edge SDK ☐ Tier 1 Bank Node ☐ Tier 2 federation |
| Status | ☐ Draft ☐ Approved ☐ Under review |

---

## 1. Description of the processing

**Nature.** Veritas trains a fraud-detection model using **federated learning**.
The model is brought to the data; the data is never brought to the model. Two
aggregation hops:

- **Device → Bank (Tier 0):** the customer's on-device model trains locally on
  device-side behavioural events; only a **DP-protected, secure-aggregated**
  update leaves the device. Neither the bank nor Veritas can read an individual
  device's update — only the aggregate. *Raw device data never leaves the
  phone.*
- **Bank → Global (Tier 2):** the in-tenancy node trains on the bank's payment /
  account data and contributes a **DP-clipped, noised model delta** to the
  control plane. *Raw bank data never leaves the bank's tenancy.*

**What is processed where:**

| Data category | Where it is processed | Does it leave? |
|---|---|---|
| Customer transaction / account data | inside the bank's tenancy (read-only via one scoped role) | **No** |
| Device behavioural-biometric events | on the customer's device | **No** |
| Model gradients / deltas | computed locally, DP-applied, then transmitted | Only the **DP-protected delta** leaves |
| Attestation quotes / round metadata | node → control plane | Yes (no personal data) |

**The trust boundary is crossed by exactly three things, none of them personal
data:** DP-protected model deltas, signed attestations/provenance records, and
control messages (round start, model version).

**Scope of processing:** ‹data sources mapped in `feature_map.yaml` — e.g.
warehouse tables, ISO 20022 pacs.008 feed›. **Volume / frequency:** ‹txns per
day; federation round cadence›. **Retention:** raw data per the bank's existing
policy (unchanged — Veritas adds no new store); model deltas are transient and
the control plane never stores per-device updates.

---

## 2. Lawful basis & necessity

| Item | Entry |
|---|---|
| Lawful basis (GDPR Art. 6) | ‹e.g. 6(1)(f) legitimate interests — fraud prevention; and/or 6(1)(c) legal obligation under PSR / AML› |
| Legitimate-interests assessment | ‹necessity + balancing test — fraud prevention vs data-subject rights› |
| Special-category data (Art. 9) | ‹typically none; confirm behavioural-biometric handling if Tier 0 in scope› |
| Why processing is necessary | Cross-institution fraud is structurally invisible to any single bank; FL is the only architecture that shares the *signal* without sharing the *data*. |
| Why proportionate | Data minimisation by construction — only model math leaves; DP bounds any individual's contribution. |

---

## 3. Data minimisation & privacy-by-design

- **No data pooling.** Federated learning replaces the forbidden move (pooling
  raw data). Bringing the model to the data is the data-minimising design.
- **Differential privacy.** Gradient clipping + Gaussian noise on every update
  (device→bank and bank→global) so individual records cannot be reconstructed
  from a delta. Per-tenant ε budget: ‹value›.
- **Secure aggregation (Tier 0).** The bank/Veritas see only the aggregate of
  device updates, never a single device's gradient.
- **Outbound-only, read-only.** The node reads bank data via **one read-only
  role** scoped to agreed tables/topics; it opens no inbound ports and creates
  no export job.
- **TEE isolation + attestation.** The node runs in a confidential-computing
  enclave; remote attestation proves the exact signed image is running and
  cannot exfiltrate data. *(Reference build uses a software attestation stub;
  production uses hardware Nitro / SEV-SNP / TDX quotes.)*

---

## 4. Evidence: `customerRecordsTransmitted: 0`

The cornerstone evidence for this DPIA is that **zero customer records are
transmitted**. This is verifiable, not asserted:

- The tenant console exposes `customerRecordsTransmitted: 0` from
  `GET /v1/tenants/{tenantId}/state` (`protocol/PROTOCOL.md`), shown as a proof
  badge in the console.
- The egress surface is restricted by construction to deltas + attestations +
  control messages — there is no API path that emits a customer record.
- The **Merkle transparency log** (`/v1/transparency/*`) records each round's
  contributors, rejected members, model hash, and attestation quotes; the bank
  can independently verify any record against the signed root
  (`/v1/transparency/proof/{seq}`).
- TEE remote attestation (production) cryptographically demonstrates the running
  image cannot exfiltrate data.

Attach: ‹console screenshot of the `customerRecordsTransmitted: 0` badge›,
‹a transparency-root verification›, ‹attestation quote sample›.

---

## 5. Risks to data subjects & mitigations

| Risk | Likelihood / impact | Mitigation | Residual |
|---|---|---|---|
| Reconstruction of an individual from a model delta | ‹L/I› | Differential privacy (clip + noise) on every update; secure aggregation at Tier 0 | ‹low› |
| Membership inference | ‹L/I› | DP budget; aggregation across many examples; deltas not raw gradients | ‹low› |
| Poisoned / malicious member corrupts the model | ‹L/I› | **Multi-Krum** robust aggregation rejects Byzantine updates (`n_byzantine=1`); rejection logged + `attack_detected` | ‹low› |
| Unauthorised access to the node | ‹L/I› | Ed25519/JWT identity; outbound-only; read-only scoped role; enclave isolation | ‹low› |
| Operator/insider exfiltration | ‹L/I› | TEE isolates the workload even from the bank's privileged operators; attestation proves the image | ‹low (prod)› |
| Tampering with provenance / audit trail | ‹L/I› | Signed append-only Merkle log; optional public anchor | ‹low› |
| Cross-tenant data leakage | ‹L/I› | Per-tenant node + per-tenant console; no cross-bank data path exists | ‹low› |
| Data residency / transfer | ‹L/I› | UK-region control plane; node runs in the bank's chosen region; data pinned by construction | ‹low› |

---

## 6. Data-subject rights & transfers

| Item | Entry |
|---|---|
| Transparency / privacy notice | ‹how customers are informed; Tier 0 in-app disclosure if applicable› |
| Access / erasure | Raw data stays in the bank's systems under existing processes; the model retains no identifiable records, so DSARs resolve against the bank's existing stores. ‹confirm› |
| International transfers | None of personal data — only model deltas leave the tenancy. Control plane in UK region. ‹confirm› |
| Automated decision-making (Art. 22) | ‹state whether scores auto-action (allow/hold/step-up/freeze) and the human-review path; the analyst console provides triage + override› |
| Sub-processors | ‹Veritas entity; explanation layer (FLock sovereign inference) runs on the local node / sovereign endpoint — explanations never include data that left the bank› |

---

## 7. Consultation, sign-off & review

| Item | Entry |
|---|---|
| Stakeholders consulted | ‹fraud team, InfoSec, legal, DPO› |
| Prior consultation with ICO required? | ‹yes/no + rationale — only if high residual risk remains› |
| DPO opinion | ‹…› |
| Decision | ☐ Proceed ☐ Proceed with conditions ☐ Do not proceed |
| Conditions / actions | ‹…› |
| Review date | ‹date / trigger — e.g. new feature map, ε change, TEE backend change› |

Signed: ‹DPO› — ‹date›
