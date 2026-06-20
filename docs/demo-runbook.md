# Veritas — demo runbook

Operational guide for running the live demo. Read this before every rehearsal and on stage. All randomness is seeded, so every run is reproducible from a reset.

---

## Prerequisites

- **Python 3.14** virtualenv in `core/` (`.venv` already present), with `core` installed editable: `pip install -e ".[dev]"`.
- **Node 26** for `web/` (and for the `mock/` fallback server).
- **MiniMax key** present in `web/.env.local` as `MINIMAX_API_KEY` (used server-side only by the `/api/explain` BFF route — never in the client bundle). The inspector falls back to a canned explanation if the key is missing, so the demo still runs without it.
- Confirm `web/.env.local` also has `NEXT_PUBLIC_API_BASE`, `MINIMAX_BASE_URL`, and `MINIMAX_MODEL`.

---

## Start order

**1. Start core (terminal 1):**

```bash
cd /Users/tom/Veritas/core
source .venv/bin/activate
uvicorn server.app:app --port 8000
```

Verify it is alive before moving on:

```bash
curl -s localhost:8000/state | head -c 120
```

You should get JSON with `"round":0`.

**2. Start web (terminal 2)** — point it at the live core:

```bash
cd /Users/tom/Veritas/web
NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev
```

Open **http://localhost:3000**. The dashboard should load with the `customerRecordsTransmitted: 0` badge and both panels showing a calm baseline.

> Tip: set `NEXT_PUBLIC_API_BASE` in `web/.env.local` instead of inline so you don't have to remember it. Inline just makes the source obvious on stage.

---

## Click sequence (the demo)

Mirrors the pitch script. Do a silent dry-run of this before going live.

1. **Show the baseline.** Point at the `customerRecordsTransmitted: 0` badge and the two side-by-side panels (left = siloed / today, right = federated / Veritas).
2. **Inject scam campaign** — click it. The mule campaign hits both worlds.
3. **Auto-run** — toggle it on (steps a federated round every ~1.2s). Narrate: siloed (left) stays red; federated (right) greens out within a few rounds.
4. **Inspect a flagged account** on the federated side → shows label + confidence + indicators, plus the live plain-English MiniMax explanation of *why* it's a mule.
5. **Inject malicious member** — click it. Watch the targeted bank chip flag and the `attack_detected` banner appear: Multi-Krum rejects the poisoned update; the federated model stays clean.
6. **Land the numbers.** Let auto-run reach ~6 rounds, then read the hero counters: federated 0.955 vs siloed 0.637, ~£295,800 prevented, ~1,160 victims spared, hours vs days. Close on sovereignty.

**Target end-state numbers** (after reset → campaign → malicious member bank0 → 6 rounds): federated detection ~0.955, siloed ~0.637; federated 2,029 victims / £517k lost; siloed 3,189 victims / £813k lost; £295,800 prevented; 0 records transmitted.

---

## Reset between rehearsals

The run is fully seeded and reproducible. To return to a clean baseline:

- Click the **Reset** button in the UI, **or**
- Hit the endpoint directly:

```bash
curl -s -X POST localhost:8000/sim/reset | head -c 80
```

Either restores `round: 0`, no campaign, no attack. Re-run the click sequence from step 1. No server restart needed between rehearsals.

---

## Fallback plan (if core misbehaves)

A standalone mock server reproduces the same contract and scripted dynamics. Point the web at it instead — the UI cannot tell the difference.

**1. Start the mock (terminal 3):**

```bash
cd /Users/tom/Veritas/mock
npm i   # first time only
npm start   # serves on :8001
```

**2. Repoint the web at the mock:**

```bash
cd /Users/tom/Veritas/web
NEXT_PUBLIC_API_BASE=http://localhost:8001 npm run dev
```

The same click sequence works against the mock: inject campaign → auto-run → siloed red, federated green → inject malicious member → `attack_detected`. The hero gap still tells the story.

---

## "If it breaks on stage" — 10-second recovery

1. **UI frozen / counters stuck?** Click **Reset** (or `curl -X POST localhost:8000/sim/reset`), then re-run from "Inject scam campaign". ~5s.
2. **Core erroring or unresponsive?** In the web terminal, Ctrl-C and restart with `NEXT_PUBLIC_API_BASE=http://localhost:8001 npm run dev` (mock already running as a warm spare). Refresh the browser. ~10s.
3. **Keep the mock running in terminal 3 throughout** so the failover is a single env-var swap, not a cold start.
4. **MiniMax slow/down?** The inspector falls back to the canned explanation automatically — keep narrating, don't wait on it.
