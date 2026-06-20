# GNN-Backed WOW Demo Design

**Date:** 2026-06-20
**Branch:** `gnn-backed-wow-demo`
**Base:** `main` at `a6fed35`

## Goal

Make the million-customer WOW section technically honest by tying its headline metrics to the new federated GNN benchmark instead of presenting the canvas-only contagion model as the technical result.

## Design

The browser canvas remains a scaled visualization of cross-bank mule spread and containment. It is not the source of truth for model performance. The source of truth becomes a `gnnBenchmark` telemetry payload carried on the existing mock `/state` and tenant-state responses.

The payload represents the deterministic Python benchmark in `core/experiments/federated_gnn.py` using its default configuration:

- `n_banks = 4`
- `n_accounts = 700`
- `n_campaigns = 8`
- `rounds = 25`
- `siloed_recall = 0.5142328861369109`
- `federated_recall = 0.8068455452356381`
- `siloed_auc = 0.7455418718497198`
- `federated_auc = 0.9332588197002297`
- `lift = 0.29261265909872725`

The dashboard will use those values to label the stage as a federated GNN mule-graph benchmark. Copy should say "visualized over 1,000,000 simulated customers" rather than "GNN over 1,000,000 customers" because the GNN benchmark itself runs over the current synthetic graph size.

## Product Behavior

The stage headline changes from a generic contagion claim to a GNN-specific claim:

`Cross-bank mule rings are a graph problem. Veritas trains the graph without moving records.`

The top metrics show:

- Siloed GNN recall
- Federated GNN recall
- Recall lift
- AUC lift

The existing panel stats remain visual-scale stats, but their labels must not imply they are directly measured by the GNN. Use labels like `visual exposure`, `visual containment`, and `mule corridors`.

The technical proof strip changes its aggregation language from generic `Multi-Krum aggregate` to `DP GNN delta -> Multi-Krum -> VSA proof`, and its metrics show:

- current round
- records moved, still `0`
- GNN recall delta

## Architecture

Add a small shared TypeScript contract for GNN telemetry in `web/src/lib/types.ts`. The mock server will expose the same shape through `/state` and `/v1/tenants/:tid/state`.

Add `mock/gnnBenchmark.js` as a deterministic fixture with the benchmark values and round interpolation. This fixture is intentionally separate from `mock/scenario.js` so the normal race counters stay independent from the GNN benchmark story.

Add `web/src/lib/gnnTelemetry.ts` for pure formatting and fallback behavior. The UI consumes this helper so missing telemetry degrades to clear fallback labels instead of crashing.

Do not run Python from the Node mock server at request time. That would make the demo slow and environment-dependent. The fixture must cite the Python experiment path and seed so it is auditable.

## Testing

Use test-first changes:

- `web/src/lib/gnnTelemetry.test.ts` proves recall/AUC lift formatting and fallback behavior.
- `web/src/lib/contagion.test.ts` proves `computeContagionFrame` can consume GNN recall as its detection source.
- `core/tests/test_federated_gnn.py` gets a small assertion that the benchmark values remain stable enough for the committed mock telemetry.
- `node --check` validates the mock server modules.

## Non-Goals

- Do not expose a production GNN API.
- Do not make the browser run the GNN.
- Do not replace the existing race panels.
- Do not claim the GNN benchmark uses one million real or synthetic graph nodes.
