import type { GnnBenchmark } from "./types";

export type { GnnBenchmark };

export interface GnnBenchmarkSummary {
  available: boolean;
  siloedRecall: number;
  federatedRecall: number;
  recallLift: number;
  aucLift: number;
  currentFederatedRecall: number;
  currentRecallLift: number;
  siloedRecallLabel: string;
  federatedRecallLabel: string;
  recallLiftLabel: string;
  aucLiftLabel: string;
  currentFederatedRecallLabel: string;
  currentRecallLiftLabel: string;
  sourceLabel: string;
  graphLabel: string;
  proofLabel: string;
}

function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatPointLift(value: number): string {
  const points = value * 100;
  const sign = points >= 0 ? "+" : "";
  return `${sign}${points.toFixed(1)} pts`;
}

export function summarizeGnnBenchmark(
  benchmark: GnnBenchmark | undefined,
): GnnBenchmarkSummary {
  if (!benchmark) {
    return {
      available: false,
      siloedRecall: 0,
      federatedRecall: 0,
      recallLift: 0,
      aucLift: 0,
      currentFederatedRecall: 0,
      currentRecallLift: 0,
      siloedRecallLabel: "GNN pending",
      federatedRecallLabel: "GNN pending",
      recallLiftLabel: "+0.0 pts",
      aucLiftLabel: "+0.0 pts",
      currentFederatedRecallLabel: "GNN pending",
      currentRecallLiftLabel: "+0.0 pts",
      sourceLabel: "GNN benchmark unavailable",
      graphLabel: "GNN graph unavailable",
      proofLabel: "GNN proof unavailable",
    };
  }

  const recallLift = benchmark.metrics.federatedRecall - benchmark.metrics.siloedRecall;
  const aucLift = benchmark.metrics.federatedAuc - benchmark.metrics.siloedAuc;
  const currentFederatedRecall =
    benchmark.current?.federatedRecall ?? benchmark.metrics.federatedRecall;
  const currentRecallLift = benchmark.current?.recallLift ?? recallLift;

  return {
    available: true,
    siloedRecall: benchmark.metrics.siloedRecall,
    federatedRecall: benchmark.metrics.federatedRecall,
    recallLift,
    aucLift,
    currentFederatedRecall,
    currentRecallLift,
    siloedRecallLabel: formatPct(benchmark.metrics.siloedRecall),
    federatedRecallLabel: formatPct(benchmark.metrics.federatedRecall),
    recallLiftLabel: formatPointLift(recallLift),
    aucLiftLabel: formatPointLift(aucLift),
    currentFederatedRecallLabel: formatPct(currentFederatedRecall),
    currentRecallLiftLabel: formatPointLift(currentRecallLift),
    sourceLabel: `${benchmark.source.path} seed ${benchmark.source.seed}`,
    graphLabel: `${benchmark.graph.banks} banks / ${benchmark.graph.accounts} accounts / ${benchmark.graph.campaigns} mule campaigns`,
    proofLabel: `${benchmark.training.privacy} -> ${benchmark.training.aggregation} -> ${benchmark.training.verification}`,
  };
}
