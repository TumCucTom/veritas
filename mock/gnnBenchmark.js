const FINAL = {
  siloedRecall: 0.5142328861369109,
  federatedRecall: 0.8068455452356381,
  siloedAuc: 0.7455418718497198,
  federatedAuc: 0.9332588197002297,
};

const SOURCE = {
  path: "core/experiments/federated_gnn.py",
  seed: 0,
  generatedAt: "2026-06-20T00:00:00.000Z",
};

const GRAPH = {
  banks: 4,
  accounts: 700,
  campaigns: 8,
  crossBankEdges: 1185,
  alertBudgetPct: 15,
};

const TRAINING = {
  rounds: 25,
  aggregation: "Multi-Krum",
  privacy: "DP clip/noise",
  verification: "VSA bounded-norm proof",
};

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function easeOut(value) {
  return 1 - Math.pow(1 - clamp01(value), 3);
}

export function gnnBenchmark(round = 0, campaignActive = false) {
  const progress = campaignActive ? easeOut(round / TRAINING.rounds) : 0;
  return {
    source: SOURCE,
    graph: GRAPH,
    training: TRAINING,
    metrics: FINAL,
    current: {
      round: Math.max(0, round),
      progress,
      siloedRecall: FINAL.siloedRecall,
      federatedRecall: 0.36 + (FINAL.federatedRecall - 0.36) * progress,
      recallLift: (FINAL.federatedRecall - FINAL.siloedRecall) * progress,
    },
  };
}
