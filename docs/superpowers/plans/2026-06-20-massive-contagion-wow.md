# Massive Contagion WOW Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a million-customer contagion visual and a compact technical proof strip to the Veritas demo.

**Architecture:** Pure deterministic simulation logic lives in `web/src/lib/contagion.ts` with tests. React components render that model with canvas and existing Veritas state. The backend contract remains unchanged.

**Tech Stack:** Next.js 16 App Router, React 19 client components, TypeScript, Canvas 2D typed arrays, Vitest for web logic tests.

## Global Constraints

- Branch is `massive-contagion-wow`, created from latest `main` at `bb23b75`.
- Do not add external visualization libraries unless native canvas cannot meet the acceptance criteria.
- Copy must say "simulated customers" and must not imply real customer data.
- Respect `prefers-reduced-motion`.
- Keep existing race panels and measured counters below the new stage.
- Verify with `npm run lint`, `npm run build`, and targeted tests.

---

### Task 1: Contagion Model

**Files:**
- Create: `web/src/lib/contagion.ts`
- Create: `web/src/lib/contagion.test.ts`
- Modify: `web/package.json`
- Modify: `web/package-lock.json`

**Interfaces:**
- Produces: `LOGICAL_CUSTOMERS`, `VISIBLE_POINTS`, `computeContagionFrame(input: ContagionInput): ContagionFrame`
- Consumes: none

- [ ] Add Vitest to `web` dev dependencies.
- [ ] Write failing tests for deterministic output, logical scale, siloed spread, and federated suppression.
- [ ] Run `npx vitest run src/lib/contagion.test.ts` and verify the tests fail because `contagion.ts` does not exist.
- [ ] Implement the deterministic model with typed-array friendly point data and aggregate counts.
- [ ] Run the targeted Vitest test and verify it passes.

### Task 2: Canvas Stage And Proof Strip

**Files:**
- Create: `web/src/components/MassiveContagionStage.tsx`
- Create: `web/src/components/TechnicalProofStrip.tsx`
- Modify: `web/src/app/page.tsx`
- Modify: `web/src/app/globals.css`

**Interfaces:**
- Consumes: `computeContagionFrame(input: ContagionInput): ContagionFrame`
- Produces: visible demo section and proof rail driven by existing state

- [ ] Add `MassiveContagionStage` as a client component using two canvases and existing `useVeritas` state.
- [ ] Draw deterministic point fields, bank community bands, red spread, and green suppression.
- [ ] Add reduced-motion snapshot behavior.
- [ ] Add `TechnicalProofStrip` showing round, records transmitted, DP/Multi-Krum/provenance state.
- [ ] Insert both components in `page.tsx` above the existing race panels.
- [ ] Run `npm run build` to verify TypeScript and Next App Router boundaries.

### Task 3: Demo Polish And Verification

**Files:**
- Modify: `web/src/components/ProvenancePanel.tsx`
- Modify: `web/src/lib/controlPlaneStore.tsx`
- Modify: `web/src/components/GovernancePanel.tsx`
- Modify: `mock/server.js` if `/provenance` still 404s

**Interfaces:**
- Consumes: existing state stores and mock API
- Produces: clean lint and console behavior

- [ ] Fix synchronous setState-in-effect lint errors by deferring initial async calls through an inner async function or transition-safe pattern accepted by the lint rule.
- [ ] Remove the unused `controlPlane` import.
- [ ] Add mock `/provenance` only if missing.
- [ ] Run `npm run lint`.
- [ ] Run `npm run build`.
- [ ] Run `core/.venv/bin/python -m pytest`, `controlplane/.venv/bin/python -m pytest`, `node/.venv/bin/python -m pytest`, and `edge-sdk` tests.
- [ ] Start mock and web, inspect the page in browser, verify no blank canvases and no console errors for the new stage.
