# Veritas — 3-minute recording storyboard

A shot-list for a clean 3-minute screen recording. The recording itself is a human step — this is the storyboard to follow. Pair it with `docs/demo-runbook.md` (start order + click sequence) and `docs/pitch.md` (the narration).

---

## Before you hit record

- Bring the stack up (runbook **Path A**) and do one full silent dry-run so the green wave and counters are seeded and predictable. **Reset** to a clean baseline (`round: 0`) just before recording.
- Browser at **1920×1080**, zoom 100%, full-screen the dashboard (hide tabs/bookmarks/notifications). Dark theme as in the demo shots.
- Record at 1080p/60fps. Mic check; one clean take per scene, or narrate live over a single continuous pass.
- Mouse: move deliberately and pause on each target ~1s before clicking so the cut reads clearly.
- If anything wobbles, prefer a clean re-take from the last reset over editing around it.

---

## Shot list (scenes → action → narrate → callout)

| # | ~Time | Scene | Click / action | Narrate (from pitch.md) | On-screen callout |
|---|---|---|---|---|---|
| 1 | 0:00–0:20 | **Title** | Hold on title card or the calm dashboard. | "One bank detects, every bank is immunised — without a single customer record leaving the building." | Lower-third: *Veritas — federated fraud intelligence for UK banks* |
| 2 | 0:20–0:40 | **Baseline** (`01-initial.png`) | Cursor circles the `customerRecordsTransmitted: 0` badge, then the two panels. | The hook: APP fraud, PSR 50/50 liability, data-can't-be-pooled. Only gradients move — clipped + noised. | Highlight badge **`records transmitted: 0`**; label panels **TODAY / VERITAS** |
| 3 | 0:40–0:55 | **Inject campaign** (`04-divergence-round2.png`) | Click **Inject scam campaign**. | A coordinated mule campaign opens safe-account mules across all banks at once — it hits both worlds. | Callout: *Mule campaign injected — both regimes* |
| 4 | 0:55–1:40 | **The race** (`02-race-round6.png`) | Toggle **Auto-run**; let rounds advance. | Siloed (left) stays red; federated (right) greens out as the round propagates the mule signature — the immunity wave overtakes the contagion. | Split callouts: **SILOED — still red** / **FEDERATED — immunised** |
| 5 | 1:40–2:10 | **Inspect a mule** (`03-inspector-provenance.png`) | Click a flagged account on the federated side; let the explanation render. | The model says fraud, high confidence — and here's the plain-English *why* a fraud analyst can act on. | Callout: *Live explanation — a "why," not just a score* |
| 6 | 2:10–2:40 | **Malicious member** (`03-inspector-provenance.png`, provenance ledger) | Click **Inject malicious member**; pan to the `attack_detected` banner + provenance ledger. | A member turns malicious to whitelist its own mules. Multi-Krum rejects it; the provenance trail records who was rejected. Model stays clean. | Callout: **Multi-Krum: poisoned update REJECTED**; highlight the rejected row |
| 7 | 2:40–3:00 | **Numbers + close** (`05-race-persistent-contrast.png`) | Let auto-run reach ~6 rounds; hold on the hero counters. | Federated 0.955 vs siloed 0.637. £295,800 prevented. 1,160 fewer victims. Hours, not days. Zero records left an institution. Real `flock-sdk`, UK-controlled. | Hero card: **0.955 vs 0.637 · £295,800 prevented · 0 records** |

---

## Narration notes

- Speak it, don't read it. The screen does most of the work — narrate what the viewer is already watching.
- Keep total runtime ≤ 3:00. If you run long, trim scene 4 (the race) — the green-overtakes-red moment lands in the first few rounds.
- Honesty guardrails (do not overclaim): say **real `flock-sdk`** and **live distributed federation**; name **FLock FL Alliance / on-chain attestation / sovereign inference** as the *production* path, not as running here. The edge fleet is illustrative; paid inference is production-not-built.

## Optional B-roll / cutaways (only if extending past 3 min)

- GNN mule-graph stage — "cross-bank mule rings are a graph problem": siloed 0.51 → federated 0.81 recall (+56.9%).
- Stacked ensemble result: 0.761 vs 0.614 best-single.
- Provenance "verify" resolving against the Merkle transparency root (consistency proof).
- Real-data slate: ULB credit-card (284k tx, 0.17% fraud), ensemble AUPRC 0.844 → 0.884 federated; fraud-poorest bank +0.066.
