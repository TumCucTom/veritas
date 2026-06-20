// Frozen fallback for the headline federated-GNN mule-graph benchmark.
//
// These constants are NOT hand-tuned: they are the REAL output of
//   core/veritas_core/gnn_benchmark.compute_gnn_benchmark(seed=0)
// (which the live backends serve in /state). Regenerate by running that
// function and pasting metrics/graph/roundTrajectory here so the mock fallback
// can never drift from the real numbers.
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

// REAL per-round federated recall trajectory measured by the numpy GNN.
const ROUND_TRAJECTORY = [
  0.37650498796009635, 0.5432576539387685, 0.5834623323013416,
  0.6391038871689027, 0.6947454420364637, 0.6947454420364637,
  0.7086343309253527, 0.7642758857929136, 0.7789817681458548,
  0.7789817681458548, 0.7789817681458548, 0.7789817681458548,
  0.7789817681458548, 0.793687650498796, 0.7789817681458548,
  0.7789817681458548, 0.7921396628826969, 0.8068455452356381,
  0.8068455452356381, 0.8068455452356381, 0.8068455452356381,
  0.8068455452356381, 0.8068455452356381, 0.8068455452356381,
  0.8068455452356381,
];

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

// Federated recall read off the REAL trajectory, indexed/interpolated by the
// live round vs TRAINING.rounds. Mirrors gnn_benchmark.benchmark_current so the
// fallback matches the backend's live `current` block.
function federatedRecallAt(round, campaignActive) {
  const traj = ROUND_TRAJECTORY;
  if (!campaignActive || round <= 0) return traj[0];
  const pos = Math.min(round, TRAINING.rounds);
  const f = pos - 1;
  if (f <= 0) return traj[0];
  if (f >= traj.length - 1) return traj[traj.length - 1];
  const lo = Math.floor(f);
  const frac = f - lo;
  return traj[lo] * (1 - frac) + traj[lo + 1] * frac;
}

export function gnnBenchmark(round = 0, campaignActive = false) {
  const r = Math.max(0, round);
  const progress = clamp01(r / TRAINING.rounds);
  const federatedRecall = federatedRecallAt(r, campaignActive);
  return {
    source: SOURCE,
    graph: GRAPH,
    training: TRAINING,
    metrics: FINAL,
    roundTrajectory: ROUND_TRAJECTORY,
    current: {
      round: r,
      progress,
      siloedRecall: FINAL.siloedRecall,
      federatedRecall,
      recallLift: federatedRecall - FINAL.siloedRecall,
    },
  };
}
