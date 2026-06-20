# Veritas — demo runbook

The live-run script. Read it before every rehearsal and on stage. All randomness is seeded, so every run is reproducible from a reset. Target: **3 minutes**, flawless.

There are two ways to run the demo. Pick one before you walk on stage:

- **A — Real stack (preferred):** the live distributed federation — a real control plane (`:9000`) + N independent node processes running real `flock-sdk` rounds. This is the one to demo: the race is genuine FL, not a script.
- **B — Mock fallback:** a standalone mock that reproduces the same contract and scripted dynamics. The UI cannot tell the difference. Keep it warm as a spare.

---

## Prerequisites (both paths)

- **Python** virtualenvs present in `controlplane/.venv`, `node/.venv`, and `core/.venv` (each `pip install -e ".[dev]"`).
- **Node.js** for `web/` (and the `mock/` fallback server).
- **MiniMax key** in `web/.env.local` as `MINIMAX_API_KEY` (used server-side only by the `/api/explain` BFF route — never in the client bundle). The inspector falls back to a canned explanation if the key is missing, so the demo still runs without it.
- Confirm `web/.env.local` also has `NEXT_PUBLIC_API_BASE` (node), `NEXT_PUBLIC_CONTROL_PLANE` (`:9000`), `MINIMAX_BASE_URL`, and `MINIMAX_MODEL`.
- Do a full silent dry-run of the click sequence below at least once before going live.

---

## Path A — Real stack (preferred)

### A1. Bring the federation up (terminal 1)

One command stands up the control plane (`:9000`) plus 5 bank nodes (`:8100..8104`):

```bash
cd /Users/tom/veritas
VERITAS_AUTOSTART_FEDERATION=1 VERITAS_BLIND_NODE=3 VERITAS_DP_SIGMA=0.1 \
  bash deploy/run_local.sh up 5
```

What the knobs mean (say this in Q&A, not on stage):
- `VERITAS_AUTOSTART_FEDERATION=1` — nodes run their own federation loop, so rounds advance on their own.
- `VERITAS_BLIND_NODE=3` — node3 is trained campaign-blind (siloed model can't see the cross-bank campaign). The **federated** model must carry the signal in from the seeing banks. This is what makes the federated-vs-siloed lift real and honest, not a strawman.
- `VERITAS_DP_SIGMA=0.1` — the demo DP noise multiplier on the 11-dim reference model. The RDP accountant still reports the *true* ε this σ yields at `/v1/privacy`.

The script prints each URL as it becomes healthy: `control plane -> http://localhost:9000`, then `node0..node4 -> http://localhost:8100..8104`.

### A2. Approve the members (terminal 1)

Members enrol as `pending`. Approve all five so they start federating (the script prints this exact line — copy it):

```bash
for m in node0 node1 node2 node3 node4; do \
  curl -s -X POST -H "X-Admin-Key: dev-admin-key" \
  http://localhost:9000/v1/members/$m/approve; done
```

Verify the federation is live before moving on:

```bash
curl -s http://localhost:9000/health
curl -s http://localhost:8100/state | head -c 120   # expect JSON with a round number
```

### A3. Start the web console (terminal 2)

```bash
cd /Users/tom/veritas/web
npm run dev
```

`web/.env.local` already points the dashboard at the node (`NEXT_PUBLIC_API_BASE=http://localhost:8000`) and the governance/provenance views at the control plane (`NEXT_PUBLIC_CONTROL_PLANE=http://localhost:9000`). If your live node is on `:8100` rather than `:8000`, start with it inline so the source is obvious on stage:

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8100 npm run dev
```

Open the printed URL (Next dev serves on **http://localhost:3000** by default). The dashboard should load with the `customerRecordsTransmitted: 0` badge and both panels at a calm baseline — matches `docs/demo-shots/01-initial.png`.

---

## The click sequence (the demo) — ~3 minutes

Mirrors `docs/pitch.md`. Timings are a guide; the screen does the work. The reference screenshots are noted per step.

| # | Time | Action | What to say / watch | Shot |
|---|---|---|---|---|
| 1 | 0:00–0:35 | **Show the baseline.** Point at the `customerRecordsTransmitted: 0` badge and the two side-by-side panels (left = siloed/today, right = federated/Veritas). | The hook: APP fraud, PSR 50/50, the data-can't-be-pooled impossibility. Only gradients move — clipped + noised. | `01-initial.png` |
| 2 | 0:35–0:55 | **Inject scam campaign** — click it. | A coordinated mule campaign opens safe-account mules across all banks at once. It hits both worlds. | `04-divergence-round2.png` |
| 3 | 0:55–1:40 | **Auto-run** — toggle on (steps a federated round ~every 1.2s). | Siloed (left) **stays red**; federated (right) **greens out** as the round propagates the mule signature. The green immunity wave overtakes the red contagion. | `02-race-round6.png` |
| 4 | 1:40–2:10 | **Inspect a flagged account** on the federated side. | Label + confidence + indicators, plus the live plain-English explanation of *why* it's a mule (new account, rapid high-value fan-out). An investigator gets a "why," not just a score. | `03-inspector-provenance.png` |
| 5 | 2:10–2:40 | **Inject malicious member** — click it. | One member turns malicious, submits poisoned updates to whitelist its own mules. Watch the chip flag + the `attack_detected` banner: **Multi-Krum rejects it.** The provenance ledger records who was rejected. Model stays clean. | `03-inspector-provenance.png` (provenance ledger) |
| 6 | 2:40–3:00 | **Land the numbers.** Let auto-run reach ~6 rounds, read the hero counters, close on sovereignty. | Federated **0.955** vs siloed **0.637**; **~£295,800 prevented**, **~1,160 victims spared**, hours vs days, **0 records transmitted**. Real `flock-sdk`; UK-controlled; nothing left any silo. | `05-race-persistent-contrast.png` |

**Target end-state** (reset → campaign → malicious member node0 → ~6 rounds): federated detection ~0.955, siloed ~0.637; federated 2,029 victims / £517k lost; siloed 3,189 victims / £813k lost; £295,800 prevented; 0 records transmitted.

**Optional depth (Q&A, not the 3-min run):** the GNN mule-graph stage shows siloed 0.51 → federated 0.81 recall (+56.9%); the stacked ensemble scores 0.761 vs 0.614 best-single. State these only if a judge probes technical depth.

---

## Reset between runs

Fully seeded and reproducible. To return to a clean baseline (no server restart needed):

- Click the **Reset** button in the UI, **or**
- Hit the endpoint directly:

```bash
curl -s -X POST http://localhost:8100/sim/reset | head -c 80   # node sim reset
```

Either restores `round: 0`, no campaign, no attack. Re-run the click sequence from step 1.

> If you teardown the whole stack between rehearsals: `bash deploy/run_local.sh down`, then re-run A1+A2. A simple `sim/reset` (above) is faster and is what you want between back-to-back runs.

---

## Path B — Mock fallback (warm spare)

A standalone mock reproduces the same contract and scripted dynamics. Keep it running throughout so failover is a single env-var swap, not a cold start.

**B1. Start the mock (terminal 3):**

```bash
cd /Users/tom/veritas/mock
npm i      # first time only
npm start  # serves on :8001
```

**B2. Point the web at the mock:**

```bash
cd /Users/tom/veritas/web
NEXT_PUBLIC_API_BASE=http://localhost:8001 npm run dev
```

The same click sequence works against the mock: inject campaign → auto-run → siloed red, federated green → inject malicious member → `attack_detected`. The hero gap still tells the story; the UI cannot tell the difference.

---

## "If X breaks, do Y" — fast recovery

| If… | Then… | Time |
|---|---|---|
| UI frozen / counters stuck | Click **Reset** (or `curl -X POST http://localhost:8100/sim/reset`), re-run from "Inject scam campaign". | ~5s |
| A node won't go healthy in A1 | Check `deploy/run_local.sh down`, re-run A1. If one node still fails, drop to `up 4` (or fewer) — the race still tells the story. | ~30s |
| Members still `pending` (no rounds advancing) | Re-run the A2 approve loop; confirm `X-Admin-Key: dev-admin-key` matches `VERITAS_ADMIN_KEY`. | ~10s |
| Control plane / nodes erroring or unresponsive | In the web terminal, Ctrl-C and restart with `NEXT_PUBLIC_API_BASE=http://localhost:8001 npm run dev` (mock = warm spare). Refresh the browser. | ~10s |
| MiniMax slow / down | The inspector falls back to the canned explanation automatically. Keep narrating; don't wait on it. | 0s |
| Provenance "verify" fails to resolve | Skip the click; narrate the Merkle transparency log + consistency proofs verbally. The ledger still shows contributors/rejected. | 0s |

**Golden rule on stage:** keep the mock (Path B) running in terminal 3 the entire time, so any real-stack wobble is one env-var swap away from a clean recovery.
