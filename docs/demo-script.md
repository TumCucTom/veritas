# Veritas — 2‑Minute Demo Script

Single take, ~120 seconds. Three quick clicks (**Auto‑run**, **Inject malicious member**,
and a **drag** to orbit the 3D embedding); the rest is scroll + narrate. Read it close to verbatim.

**Before you start:** site at `localhost:3000` on the **baseline** (round 00, "records transmitted 0", panels calm).

---

## The run

### Act 1 — The race · 0:00–0:50

| Time | Do | Say |
|------|----|-----|
| **0:00** | Sit on the baseline. Point at **"customer records transmitted: 0"**. | "Scam fraud crosses banks in minutes — but the data that would catch it can't be pooled. So every bank fights blind, alone." |
| **0:10** | Click **Auto‑run**. | "Veritas trains one model across all of them — sharing *intelligence*, never data. Watch a mule campaign hit." |
| **0:18** | Point at the **hero** (Time to detect). | "Alone, banks catch it in **days**. Federated — in **hours**." |
| **0:28** | Point at the **two contagion clouds**, then the bank strip. | "Same million customers. Siloed goes **red** as fraud spreads bank to bank; Veritas stays **green** and contains it. **96% caught versus 59%** — every bank lifts, even the one blind to the campaign." |
| **0:42** | Click **Inject malicious member**. | "And when a bank turns malicious and submits a *poisoned* update — Multi‑Krum **rejects it**. The model stays clean, and the rejection is logged." |

### Act 2 — Under the hood · 0:50–1:40

Scroll to **"Under the hood · live from the models."**

| Time | Do | Say |
|------|----|-----|
| **0:50** | Open the section. | "And none of this is a mock‑up — it's all precomputed from the **real** models." |
| **0:55** | **Drag to orbit** the **Embedding Atlas**; scrub a round. | "The model's learned space, in 3D. As federation proceeds, the **scam cluster physically pulls apart** from honest traffic — a separation no single bank reaches alone." |
| **1:08** | Point at **Ensemble Stack**. | "Five models — logistic, MLP, embeddings, a GRU, federated boosted trees — each catch a different fraud typology. A meta‑learner stacks them: **0.75 recall, beating 0.56** for the best single model." |
| **1:20** | Point at **Mule Graph**. | "The federated GNN surfaces the **cross‑bank mule rings** — the rings no single bank could see." |
| **1:30** | Gesture to **Privacy Budget** + **Zero‑Knowledge proof**. | "Privacy is **measured, not assumed** — ε ≈ 4.2 under a formal budget — and we can prove an update is in‑bounds **without ever seeing it**." |

### Act 3 — Trust & close · 1:40–2:00

| Time | Do | Say |
|------|----|-----|
| **1:40** | Scroll to **Governance** (badge: **CONTROL PLANE · LIVE**). | "Live control plane: every node **attested**, every model change in a **tamper‑evident log**, and the line under all of it — **zero customer records ever moved**. UK‑controlled, on infrastructure the banks own." |
| **1:52** | — | "Hours not days. **£847k saved** this campaign. Collective immunity no bank could build alone. **That's Veritas.**" |

**The three numbers to land:** **96% vs 59%** · **hours vs days** · **0 records moved.**

---

## Quick Q&A (one‑liners)

- **Real or scripted?** Federation runs real `flock_sdk` rounds; every "Under the hood" panel is precomputed from the **real** models (real recall, real ε, real proofs).
- **Privacy strength?** Rényi‑DP accountant, **ε ≈ 4.2 at δ = 1e‑5**, under an ε = 8 budget — show the **Privacy Budget** panel.
- **Malicious bank?** Multi‑Krum robust aggregation + a zero‑knowledge norm proof — demonstrated live with *Inject malicious member*; the **Federation Pulse** and **ZK Proof** panels show the mechanics.
- **Why federated wins?** The campaign is cross‑bank; the blind node never sees it in its own books, so federation carries the signal in. Honest lift, not a strawman.
- **Deeper bench** (only if asked): GRU sequence timeline, federated GBDT histograms, secure‑aggregation pulse — all on the page; pull up whichever the question points at.

## Fallback & timing

- **If a panel looks static:** it's client‑driven — click **Auto‑run** (it starts the campaign) and the race diverges within a round.
- **Reset** for a clean baseline between runs. **Teardown:** `bash deploy/run_local.sh down` + kill the `:3000`/`:8001` processes.
- **Short on time?** Cut Act 2 to just the **Embedding Atlas** + **Ensemble** beats and you're at ~75 seconds. **More time?** Linger on any "Under the hood" panel a judge reacts to.
