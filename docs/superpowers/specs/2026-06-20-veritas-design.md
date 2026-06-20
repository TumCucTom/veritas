# Veritas — Design Spec

**Date:** 2026-06-20
**Status:** Approved (brainstorm → design). Next: implementation plan.
**Challenge:** FLock.io × UK Sovereign AI hackathon.

---

## One line

A **cross-institution federated fraud-intelligence network**: UK banks and telcos collaboratively train a shared fraud/scam-detection model **without any customer data leaving any institution** — so a new fraud campaign detected at one member inoculates the entire network within hours instead of days.

> Production target: FLock's live **FL Alliance** network. Hackathon build: real `flock-sdk` federated learning across simulated institutional nodes, locally orchestrated for interactive demo control. **Crypto-free** — no wallet, no tokens.

---

## 1. The problem (first principles)

The data that would crush UK fraud — customer transactions, account behaviour, scam reports — is exactly the data that **cannot be pooled**. Banks are legally barred (GDPR, competition law) and commercially unwilling to share raw customer data. Yet fraud is inherently **cross-institutional**: a mule network or APP-fraud ("safe account") campaign opens accounts across many banks at once. No single bank sees the whole pattern, so each detects it slowly and independently.

This is a structural impossibility that centralised AI cannot solve — pooling is the one forbidden move. **Federated learning is the only architecture that works:** bring the model to each bank's data, share only model updates, keep ownership and data local.

### Why now
Since **October 2024 the PSR mandates APP-fraud reimbursement**, split 50/50 between sending and receiving banks. Banks are now financially exposed to *other* banks' fraud and are desperate to (a) detect faster and (b) identify mule (receiving) accounts early. This creates an acute, budgeted willingness to pay.

---

## 2. Challenge alignment

| Rubric criterion | How Veritas hits it |
|---|---|
| Clear, relevant use case | Cross-bank APP/mule fraud — multi-£bn UK problem with regulatory urgency |
| Decentralised / privacy-preserving AI | Real `flock-sdk` federated learning; no raw data shared; DP on updates |
| Strong technical implementation | Federated rounds, robust aggregation, differential privacy, poisoning defence, live two-regime simulation |
| Trust, governance, security | Malicious-member (poisoning) defence; tamper-evident model provenance; "no data left the silo" attestation |
| **Credible path to adoption** | Reachable, payable institutional buyers (Cifas, UK Finance, Stop Scams UK, bank fraud teams, telcos); quantified £-prevented business case gathered as outreach evidence during the hackathon |

Primary track: **Trusted Data & AI Infrastructure** (cross-institution model training; secure AI for regulated industries). Secondary: **AI Governance, Transparency & Trust** (poisoning defence, provenance).

---

## 3. Real FLock usage (crypto-free)

| Surface | Use in Veritas | Cost |
|---|---|---|
| `flock-sdk` (federated learning) | Veritas is a real FLock `FlockModel` (`train`/`evaluate`/`aggregate`), forked from their **credit-card-fraud example** and repurposed. Rounds orchestrated locally for demo control. | Docker/local. No key, no wallet. |
| Inference API (`api.flock.io`) | **Production swap** for the explanation layer; named in the pitch as the sovereign-inference path. | API key only (not used in hackathon build). |
| Live FL Alliance network | Stated production deployment target. | Wallet + tokens — **out of scope** for the build. |

The "explain the scam/fraud" layer uses **MiniMax** (OpenAI-compatible, key in `.env` as `MINIMAX_API_KEY`), called **server-side only**. Disclosure: MiniMax is a non-UK provider used purely as a pluggable UX explainer; the sovereign core is the federated model. Production swaps MiniMax → FLock sovereign inference (one line).

---

## 4. Architecture

```
   contract/  ── OpenAPI + shared TS types + pydantic (the ONLY shared surface)
      ▲                                            ▲
 PERSON A: core/  (Python)                   PERSON B: web/  (Next.js)
 • flock-sdk FlockModel (tabular fraud)      • Split-screen race dashboard
 • N simulated bank/telco nodes              • Population particle simulation
 • robust aggregation (Multi-Krum /          • Live counters (£ / time / victims)
   trimmed-mean)                             • Bank/transaction inspector
 • differential privacy on updates           • MiniMax "explain" (server BFF route;
 • TWO regimes per round:                       key stays server-side)
     federated  vs  siloed-baseline          • builds against mock/ from minute 1
 • two attacks: fraud campaign + poisoning
 • FastAPI: REST + WebSocket event stream
```

### Monorepo
```
veritas/
├── core/        # Person A — Python: flock-sdk engine + FastAPI server
│   ├── veritas_core/   # flockmodel.py, aggregation.py, dp.py, attack.py, data.py, regimes.py
│   ├── server/         # FastAPI app exposing the contract
│   └── pyproject.toml
├── web/         # Person B — Next.js (App Router) + Tailwind + MiniMax BFF
├── contract/    # openapi.yaml + generated TS types + pydantic models
├── mock/        # mock server implementing the contract (unblocks Person B day 1)
└── docs/superpowers/specs/2026-06-20-veritas-design.md
```

`core/` and `web/` meet **only** at `contract/`. Each runs, develops, and commits independently.

---

## 5. The centrepiece demo — the two-regime race

The same injected fraud campaign hits two worlds side by side:

```
   TODAY — siloed banks            │       VERITAS — federated
   ┌────────────────────────┐      │      ┌────────────────────────┐
   │ ●●●●● 🔴 spreading…     │      │      │ ●●●●● 🟢 immunised      │
   │ bank-by-bank, blind     │      │      │ one detects → all know  │
   │ ⏱ detect: 4.2 days      │      │      │ ⏱ detect: 3 hours       │
   │ 💷 lost: £2,140,000 ▲    │      │      │ 💷 lost: £180,000        │
   │ 🧍 victims: 8,400        │      │      │ 🧍 victims: 510          │
   └────────────────────────┘      │      └────────────────────────┘
         HERO: £1.96M prevented · 34× faster detection · 7,890 victims spared
```

- **Institutional layer:** 6–10 named bank/telco nodes = the real `flock-sdk` federation.
- **Population layer:** millions of customer particles (canvas/WebGL). A customer's chance of being hit = a function of their bank's current detection rate. **Honest framing:** real federated ML + an epidemiology-style population overlay *driven by the real per-round model metrics*.
- **Siloed regime** = identical code, model updates simply **not shared** between nodes. This is a fair counterfactual, not a strawman.

### The injected attack (real fraud dynamic)
A coordinated **mule-account / "safe account" APP-fraud campaign** opens fresh accounts across multiple banks simultaneously.
- *Siloed:* each bank sees only its slice; money keeps draining through mules at other banks; losses climb on both sending and receiving banks. Red contagion sweeps the population.
- *Federated:* the moment one bank's model learns the mule signature, the federated update propagates it to every member within hours; receiving banks freeze mules early; money stays in the system. Green immunity wave overtakes the red.

### The second attack — malicious member (security beat)
One bank-node (or a Sybil) submits poisoned updates to whitelist its own mules. **Robust aggregation (Multi-Krum / trimmed-mean) detects and rejects it** → `attack_detected`; the federated model stays clean while a naive average would be corrupted. This is the real threat model for cross-org federation → lands as serious governance/security, not a toy.

---

## 6. Data & model

- **Model:** tabular fraud/mule-account classifier (small NN or logistic), reusing the `flock-sdk` credit-card-fraud example structure. Fixed-dimension weight vector → clean federated averaging + Multi-Krum + DP, and explainable on stage.
- **Data:** **PaySim** (synthetic mobile-money transactions designed for fraud research; includes transfer/cash-out fraud) partitioned non-IID across bank nodes. Injected "new MO" campaigns = synthetic fraud clusters added at demo time.
- **Explainability:** flagged transaction/account → MiniMax renders plain-English indicators ("new account, rapid high-value transfers fanning out to multiple recipients — classic mule pattern").

---

## 7. Privacy, security, governance

- **No raw data shared** — only model updates leave a node. UI surfaces `customerRecordsTransmitted: 0`.
- **Differential privacy** — gradient clipping + Gaussian noise on updates so individual records can't be reconstructed.
- **Robust aggregation** — Multi-Krum / trimmed-mean resists poisoning by malicious members.
- **Provenance** — each round's contributing nodes + validation scores recorded as a tamper-evident log (model "bill of materials"); production maps to FLock's on-chain attestation.

---

## 8. API contract (the seam)

REST + WebSocket. Shared schema as `pydantic` (core) and generated TS types (web).

**REST**
- `GET /state` → `{ round, banks[], counters }` where `counters` carries both regimes.
- `GET /banks` → per-node metrics (`detectionRate` under `federated` and `siloed`, customers, status).
- `POST /predict` `{ transaction | text }` → `{ label, confidence, indicators[] }`.
- `POST /campaign/inject` `{ typology }` → starts the fraud campaign in both regimes.
- `POST /attack/inject` `{ memberId }` → starts the poisoning attack.
- `POST /round/step` (or auto-run loop) → advances one federated round.

**WebSocket `/events`** → `round_complete · client_updated · fraud_propagated · attack_detected`.

**Two-regime fields (the new bit):** every per-bank metric and every counter is reported under `federated` and `siloed`. Hero counters: `fraudPreventedGbp`, `timeToDetectHours` (per regime), `victims` (per regime), `compCostNote`.

---

## 9. Two-person workstreams

- **Person A — `core/`:** `flock-sdk` engine, simulated nodes, robust aggregation, DP, both attacks, the two-regime round loop, FastAPI server emitting the contract.
- **Person B — `web/`:** Next.js split-screen race + population particle sim + counters + bank/transaction inspector, and the MiniMax BFF route. Builds against `mock/` from minute 1; integration = flip the base URL.

Mock-first means neither blocks the other. The MiniMax call lives entirely in Person B's server route, decoupled from the FL core.

---

## 10. Demo script (≈3 min)

1. Network of named banks; shared model; **`customerRecordsTransmitted: 0`**.
2. Inject a new mule/APP-fraud campaign → it hits both worlds.
3. **Siloed:** red contagion spreads; £ lost climbs; detection takes "days". **Federated:** one bank detects → federated round → green immunity wave sweeps the population in "hours".
4. Inspect a flagged account on the federated side; MiniMax explains *why* it's fraud.
5. **Malicious member** tries to poison the model → robust aggregation rejects it → `attack_detected`; model stays clean.
6. Close on the hero numbers (£ prevented, 34× faster, victims spared) and sovereignty: nothing left any institution; UK-owned, collectively-improving model; production runs on FLock's FL Alliance.

---

## 11. Adoption strategy (the win condition)

Gather **real institutional demand signals during the hackathon** and put a named voice on a slide.

**Targets (reachable this weekend):**
- **Cifas** — not-for-profit fraud-prevention membership org already running shared fraud databases; perfect design partner and proof the market wants collaborative fraud intelligence.
- **UK Finance** — banking trade body.
- **Stop Scams UK** — banks + telcos + tech; runs the 159 service.
- Individual **bank fraud teams** and a **telco fraud desk**.

**The ask:** 15-min call → "yes, we'd pilot this" / a quote / a willingness-to-pay signal.

**Evidence to collect:** current cross-bank fraud-sharing latency ("days"), pain quotes, willingness to pay, one named supportive voice.

**Outreach message (3 lines):**
> We've built Veritas: a privacy-preserving way for banks to share fraud intelligence — a federated model that learns across institutions so a scam detected at one bank protects customers at all of them, with **no customer data ever leaving your systems**. In simulation it cut cross-bank fraud detection from days to hours. Could we get 15 minutes to show you and hear whether this fits a real need?

---

## 12. Scope (going for everything)

**Core (must):** real `flock-sdk` federated rounds; two-regime split-screen; population particle sim; hero counters; fraud-campaign injection.
**Full (committed, time permitting in order):** robust aggregation + poisoning attack/defence; differential privacy; MiniMax explainer; provenance log; polished UI + pitch deck.

---

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `flock-sdk` orchestration too rigid for interactive demo | Use its `FlockModel` (`train`/`aggregate`) primitives, drive the round loop ourselves — still genuinely `flock-sdk`. |
| Rendering "millions" of nodes | Canvas/WebGL particle system; it's a viz over a simulated population, never DOM nodes. |
| Population overlay read as "fake" | Disclose clearly: real FL metrics drive an epidemiology-style overlay; siloed regime is the same code with sharing off. |
| MiniMax sovereignty optics | Frame as pluggable explainer; FLock sovereign inference is the production swap. |
| Two people block each other | `contract/` + `mock/` first; integrate by flipping base URL. |
| `MINIMAX_API_KEY` exposure | Server-side BFF only; never in the client bundle; `.env` git-ignored. |

---

## 14. Out of scope (YAGNI)

On-chain/testnet deployment; real bank data or partnerships; production auth/multi-tenancy; mobile-native app; anything beyond the demo + pitch + adoption evidence.
