import { describe, expect, it } from "vitest";
import {
  summarizeGnnBenchmark,
  type GnnBenchmark,
} from "./gnnTelemetry";

const benchmark: GnnBenchmark = {
  source: {
    path: "core/experiments/federated_gnn.py",
    seed: 0,
    generatedAt: "2026-06-20T00:00:00.000Z",
  },
  graph: {
    banks: 4,
    accounts: 700,
    campaigns: 8,
    crossBankEdges: 116,
    alertBudgetPct: 15,
  },
  training: {
    rounds: 25,
    aggregation: "Multi-Krum",
    privacy: "DP clip/noise",
    verification: "VSA bounded-norm proof",
  },
  metrics: {
    siloedRecall: 0.5142328861369109,
    federatedRecall: 0.8068455452356381,
    siloedAuc: 0.7455418718497198,
    federatedAuc: 0.9332588197002297,
  },
  current: {
    round: 2,
    progress: 0.221,
    siloedRecall: 0.5142328861369109,
    federatedRecall: 0.4586,
    recallLift: 0.0647,
  },
};

describe("summarizeGnnBenchmark", () => {
  it("formats the benchmark recall and auc lift", () => {
    const summary = summarizeGnnBenchmark(benchmark);

    expect(summary.available).toBe(true);
    expect(summary.siloedRecallLabel).toBe("51.4%");
    expect(summary.federatedRecallLabel).toBe("80.7%");
    expect(summary.recallLiftLabel).toBe("+29.3 pts");
    expect(summary.aucLiftLabel).toBe("+18.8 pts");
    expect(summary.currentFederatedRecallLabel).toBe("45.9%");
    expect(summary.currentRecallLiftLabel).toBe("+6.5 pts");
  });

  it("keeps source and graph assumptions visible", () => {
    const summary = summarizeGnnBenchmark(benchmark);

    expect(summary.sourceLabel).toBe("core/experiments/federated_gnn.py seed 0");
    expect(summary.graphLabel).toBe("4 banks / 700 accounts / 8 mule campaigns");
    expect(summary.proofLabel).toBe("DP clip/noise -> Multi-Krum -> VSA bounded-norm proof");
  });

  it("returns an explicit fallback when telemetry is missing", () => {
    const summary = summarizeGnnBenchmark(undefined);

    expect(summary.available).toBe(false);
    expect(summary.federatedRecallLabel).toBe("GNN pending");
    expect(summary.recallLift).toBe(0);
  });
});
