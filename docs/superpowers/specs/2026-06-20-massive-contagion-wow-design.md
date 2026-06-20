# Massive Contagion WOW Design

**Date:** 2026-06-20
**Branch:** `massive-contagion-wow`
**Base:** `main` at `bb23b75`

## Goal

Make the Veritas demo prove technical excellence while adding one memorable visual: a million-customer fraud-contagion graph where the siloed world lets fraud spread and the federated world visibly suppresses it with a green immunity wave.

## Research Basis

- Fraud propagation should behave like a network diffusion process, not random particle noise. The UI will use an independent-cascade/SIR-inspired model: compromised seed nodes attempt to activate nearby graph neighborhoods, while protected nodes are removed from the vulnerable set as detection improves. References: Kempe, Kleinberg, and Tardos on independent cascade influence diffusion (https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf) and Newman on SIR epidemics over networks (https://link.aps.org/doi/10.1103/PhysRevE.66.016128).
- Financial-fraud simulation should be framed as synthetic and privacy-preserving. The copy will reference PaySim-style synthetic financial-transaction simulation, not real customer data. Reference: PaySim paper (https://www.msc-les.org/proceedings/emss/2016/EMSS2016_249.pdf).
- Rendering must not try to create DOM nodes. The implementation will keep a logical scale of 1,000,000 customers and render an aggregated typed-array canvas/WebGL-style point field. deck.gl's performance guidance warns that millions of visible fragments can be expensive and recommends careful point sizing and aggregation for large point sets (https://deck.gl/docs/developer-guide/performance).

## Product Shape

### 1. Massive Contagion Stage

Add a full-width `MassiveContagionStage` between `HeroCounters` and the existing two race panels.

The stage has two synchronized canvases:
- **Siloed:** red fraud spreads through bank communities and cross-bank mule corridors as rounds advance.
- **Veritas:** the same red campaign starts, then a green protection wave overtakes it as federated detection rises.

The visual should feel much bigger than the existing 6,000-dot canvas:
- label the logical population as `1,000,000 simulated customers`;
- use many visible micro-points, faint community bands, and sampled cross-bank corridors;
- expose live summary numbers: exposed, protected, suppressed, and cross-bank links;
- respect reduced motion by rendering a stable final snapshot.

### 2. Technical Proof Strip

Add a compact `TechnicalProofStrip` near the stage, not as the main spectacle.

It shows the engineering proof in one line:
`local training -> DP clip/noise -> Multi-Krum aggregation -> provenance`

It must bind to existing state:
- current round;
- average federated and siloed detection;
- `customerRecordsTransmitted`;
- whether a campaign is active;
- whether a malicious update has been injected.

### 3. Demo Polish Fixes

Fix demo issues found on latest `main`:
- web lint errors from synchronous state-setting effects in `ProvenancePanel.tsx` and `controlPlaneStore.tsx`;
- unused `controlPlane` import warning in `GovernancePanel.tsx`;
- mock `/provenance` 404 if still present on latest `main`.

## Architecture

Keep all new simulation logic frontend-local and deterministic. The backends already provide the real federated state; the new graph is a visual model driven by that state, not a new source of truth.

Files:
- `web/src/lib/contagion.ts`: pure deterministic math for graph layout, infection/protection frontiers, and aggregate counts.
- `web/src/lib/contagion.test.ts`: unit tests for monotonic protection, siloed spread, deterministic output, and logical population scale.
- `web/src/components/MassiveContagionStage.tsx`: client component rendering the synchronized canvases and summary.
- `web/src/components/TechnicalProofStrip.tsx`: client component rendering the technical proof rail.
- `web/src/app/page.tsx`: inserts the new components.
- existing lint-fix files as needed.

## Constraints

- Do not add external visualization dependencies unless the native canvas implementation cannot pass build/performance checks.
- Do not remove the existing race panels; they remain the measured outcome below the new stage.
- Keep the signature visual to one section. The rest of the dashboard should remain restrained.
- No live customer data claims. Copy must say simulated customers and synthetic fraud dynamics.
- Respect `prefers-reduced-motion`.

## Acceptance Criteria

- A judge can understand the technical story in under ten seconds: no records leave institutions, local updates are privacy-filtered, aggregation rejects malicious input, and provenance is recorded.
- A judge can see the WOW story without narration: red spreads in siloed, green overtakes it in Veritas.
- The visual is deterministic for the same round/state and does not flicker randomly between renders.
- `npm run lint` and `npm run build` pass in `web/`.
- Existing Python and edge SDK tests still pass.
