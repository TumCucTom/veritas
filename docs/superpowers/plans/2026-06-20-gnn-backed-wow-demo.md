# GNN-Backed WOW Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tie the WOW stage's headline metrics and technical proof strip to the federated GNN benchmark while keeping the canvas clearly labelled as a visualization.

**Architecture:** Add a mock `gnnBenchmark` telemetry fixture sourced from `core/experiments/federated_gnn.py`, add web helpers/types to consume it, and update the stage/proof strip copy and metrics. Runtime Python execution is avoided; the fixture cites the benchmark source and seed.

**Tech Stack:** Next.js/React/TypeScript, Vitest, Express mock server, Python pytest for the benchmark guard.

## Global Constraints

- Work on branch `gnn-backed-wow-demo` in `/Users/tom/veritas/.worktrees/gnn-backed-wow-demo`.
- Do not claim the browser canvas is the GNN benchmark.
- Do not claim the GNN benchmark ran over 1,000,000 graph nodes.
- Keep `customerRecordsTransmitted` at `0`.
- Use existing CSS tokens and component style because `/Users/tom/.Codex/design/DESIGN.md` is not present.

---

### Task 1: Web Telemetry Contract

**Files:**
- Modify: `web/src/lib/types.ts`
- Create: `web/src/lib/gnnTelemetry.ts`
- Create: `web/src/lib/gnnTelemetry.test.ts`

**Interfaces:**
- Produces: `GnnBenchmark`, `summarizeGnnBenchmark(benchmark?: GnnBenchmark): GnnBenchmarkSummary`
- Consumes: benchmark values from the mock `/state` response.

- [ ] **Step 1: Write failing test**

Create `web/src/lib/gnnTelemetry.test.ts` with tests for percent formatting, lift math, and missing telemetry fallback.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/lib/gnnTelemetry.test.ts`

Expected: fail because `gnnTelemetry.ts` does not exist.

- [ ] **Step 3: Implement types and helper**

Add `GnnBenchmark` to `types.ts` and implement `summarizeGnnBenchmark`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd web && npx vitest run src/lib/gnnTelemetry.test.ts`

Expected: pass.

### Task 2: Mock GNN Telemetry

**Files:**
- Create: `mock/gnnBenchmark.js`
- Modify: `mock/scenario.js`
- Modify: `mock/server.js`

**Interfaces:**
- Produces: `gnnBenchmark(round, campaign)` object on node `/state` and tenant `/v1/tenants/:tid/state`.
- Consumes: deterministic benchmark values from `core/experiments/federated_gnn.py`.

- [ ] **Step 1: Write failing syntax/shape test**

Run a Node import smoke command that expects `gnnBenchmark` to exist and include `source.path`.

- [ ] **Step 2: Implement fixture and attach it to state**

Add the benchmark fixture and include it in both state contracts.

- [ ] **Step 3: Verify mock syntax and shape**

Run: `cd mock && node --check gnnBenchmark.js && node --check scenario.js && node --check server.js`

Expected: pass.

### Task 3: GNN-Aware Stage and Proof Strip

**Files:**
- Modify: `web/src/components/MassiveContagionStage.tsx`
- Modify: `web/src/components/TechnicalProofStrip.tsx`
- Modify: `web/src/lib/contagion.ts`
- Modify: `web/src/lib/contagion.test.ts`

**Interfaces:**
- Consumes: `state.gnnBenchmark`.
- Produces: GNN-backed headline metrics and honest canvas labels.

- [ ] **Step 1: Write failing contagion test**

Extend `web/src/lib/contagion.test.ts` to require GNN recall to drive federated containment.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd web && npx vitest run src/lib/contagion.test.ts`

Expected: fail until `computeContagionFrame` consumes GNN recall.

- [ ] **Step 3: Update implementation and components**

Use `summarizeGnnBenchmark` in the stage and proof strip. Keep canvas rendering deterministic.

- [ ] **Step 4: Verify web tests**

Run: `cd web && npx vitest run src/lib/gnnTelemetry.test.ts src/lib/contagion.test.ts`

Expected: pass.

### Task 4: Benchmark Guard and Final Verification

**Files:**
- Modify: `core/tests/test_federated_gnn.py`

**Interfaces:**
- Consumes: `experiments.federated_gnn.run`.
- Produces: a guard that the benchmark stays in the same performance range as the UI fixture.

- [ ] **Step 1: Add benchmark-range test**

Assert the default seed keeps federated recall above `0.79`, siloed recall between `0.49` and `0.54`, and federated AUC above `0.92`.

- [ ] **Step 2: Run core test**

Run: `cd core && ./.venv/bin/python -m pytest tests/test_federated_gnn.py`

Expected: pass.

- [ ] **Step 3: Run final verification**

Run:

```bash
cd web && npx vitest run src/lib/gnnTelemetry.test.ts src/lib/contagion.test.ts && npm run lint && npm run build
cd ../core && ./.venv/bin/python -m pytest tests/test_federated_gnn.py
cd ../mock && node --check gnnBenchmark.js && node --check scenario.js && node --check server.js
```

Expected: all pass.
