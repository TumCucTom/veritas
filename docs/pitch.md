# Veritas — 3-minute demo script

**Tagline:** *One bank detects, every bank is immunised — without a single customer record ever leaving the building.*

**Headline stat:** **0.955 vs 0.637** federated vs siloed detection — **£295,800 fraud prevented**, detection cut from days to hours, with **0 customer records transmitted**.

> Speak this, don't read it. Timings are a guide. The screen does most of the work — narrate what the audience is already watching.

---

## 0:00 — The hook (≈35s)

> "APP fraud — authorised push payment scams, the 'move your money to a safe account' con — is one of the UK's largest fraud categories, costing the public hundreds of millions a year. And since October 2024 the PSR makes banks **reimburse victims by law**, split 50/50 between the sending and the receiving bank.
>
> So banks are now financially on the hook for *each other's* fraud. The receiving bank — the one whose account the money drains into — pays half. The problem: a mule campaign opens accounts across many banks at once, but no single bank sees the whole pattern. Each one detects it alone, slowly. The data that would stop this can't legally be pooled.
>
> Federated learning is the one architecture that works: bring the model to the data, share only what the model learns, never the data itself. That's Veritas."

---

## 0:35 — The setup (≈20s)

> "Eight UK banks. One shared fraud model, trained across all of them. Watch the badge — **customer records transmitted: zero**. Only model gradients move, and they're clipped and noised so no individual record can be reconstructed.
>
> Two worlds, side by side. On the left, today: every bank siloed, blind to the others. On the right, Veritas: federated. Same model, same code — the only difference is whether the banks share what they learn."

---

## 0:55 — Inject the campaign (≈45s)

*Click "Inject scam campaign", then start auto-run.*

> "A coordinated mule campaign opens safe-account mules across all eight banks at once. Watch both worlds.
>
> **Siloed — left — stays red.** Each bank only sees its own slice. The money keeps draining through mules at the *other* banks, so losses climb on both the sending and the receiving side. This takes days to pin down.
>
> **Federated — right — greens out.** One bank's model learns the mule signature, the federated round propagates it to every member, and the whole network freezes those mules within hours. The green immunity wave overtakes the red contagion. That's the entire story in one screen."

---

## 1:40 — Inspect a flagged mule (≈30s)

*Open the inspector on a flagged account, federated side.*

> "Let's open one flagged account. The model says fraud, high confidence. And here's the plain-English reason a fraud analyst can act on: new account, rapid high-value transfers fanning out to many recipients — a classic mule pattern matching the active campaign. That explanation is generated live, so an investigator gets a 'why', not just a score."

---

## 2:10 — The malicious member (≈30s)

*Click "Inject malicious member".*

> "Now the real cross-org threat. One member turns malicious and submits poisoned updates — trying to drag the shared model into whitelisting *its own* mules. This is the attack that kills naive federation.
>
> Watch — **rejected.** Robust aggregation, Multi-Krum, spots the outlier update and drops it before it touches the global model. The provenance trail records who contributed and who was rejected, every round. The federated model stays clean. No single member, honest or hostile, can corrupt the network."

---

## 2:40 — The numbers + sovereignty close (≈20s)

> "The measured result. Federated detection **0.955** against siloed **0.637** — that gap is everything. **£295,800 of fraud prevented. 1,160 fewer victims. Hours to detect instead of days.** And zero customer records ever left an institution.
>
> This is a real `flock-sdk` model. It's UK-controlled, privacy-preserving, poisoning-resistant, and every contribution is provenance-anchored. Nothing leaves the silo — the data stays sovereign, only the intelligence is shared. In production it runs on FLock's FL Alliance.
>
> Veritas: one bank detects, every bank is immunised. Thank you."

---

## Hero numbers (ground truth — measured live)

| Metric | Federated (Veritas) | Siloed (today) |
|---|---|---|
| Detection rate (avg) | **0.955** | 0.637 |
| Victims | 2,029 | 3,189 |
| Lost | £517k | £813k |
| Time to detect | ~6 hours | ~101 hours (days) |
| Fraud prevented | **£295,800** | — |
| Customer records transmitted | **0** | n/a |

*Sequence that produces these: reset → inject campaign → inject malicious member (bank0) → 6 federated rounds. All randomness is seeded, so the run is reproducible.*

**Victims spared:** 3,189 − 2,029 = **1,160**. **Loss avoided:** £813k − £517k = **~£296k** (reported as £295,800 prevented).

---

## The three beats, in one line each

1. **Privacy** — only gradients move, clipped + Gaussian-noised (differential privacy); the badge reads `customerRecordsTransmitted: 0`.
2. **Speed** — federated propagation immunises the whole network in hours; siloed takes days.
3. **Trust** — Multi-Krum rejects the poisoned member; the provenance trail records every contributor.
