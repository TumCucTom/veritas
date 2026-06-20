export interface Detection {
  federated: number;
  siloed: number;
}
export interface Bank {
  id: string;
  name: string;
  customers: number;
  detection: Detection;
  poisoned: boolean;
}
export interface RegimeCounters {
  fraudPreventedGbp: number;
  timeToDetectHours: number;
  victims: number;
  lostGbp: number;
}
export interface Counters {
  federated: RegimeCounters;
  siloed: RegimeCounters;
}
export interface GnnBenchmark {
  source: {
    path: string;
    seed: number;
    generatedAt: string;
  };
  graph: {
    banks: number;
    accounts: number;
    campaigns: number;
    crossBankEdges: number;
    alertBudgetPct: number;
  };
  training: {
    rounds: number;
    aggregation: string;
    privacy: string;
    verification: string;
  };
  metrics: {
    siloedRecall: number;
    federatedRecall: number;
    siloedAuc: number;
    federatedAuc: number;
  };
  current?: {
    round: number;
    progress: number;
    siloedRecall: number;
    federatedRecall: number;
    recallLift: number;
  };
}
export interface State {
  round: number;
  running: boolean;
  banks: Bank[];
  counters: Counters;
  campaignActive: boolean;
  attackActive: boolean;
  customerRecordsTransmitted: number;
  gnnBenchmark?: GnnBenchmark;
}
export interface Provenance {
  round: number;
  contributors: string[];
  rejected: string[];
  globalRecall: number;
}
export type Regime = "federated" | "siloed";

// Scoring contract for the engine /predict route. The transaction is the
// whitelisted numeric feature vector (camelCase, mirrors the contract).
export interface PredictRequest {
  transaction: Record<string, number>;
}
export interface PredictResponse {
  label: string;
  confidence: number;
  indicators: string[];
}

export type VeritasEvent =
  | { type: "round_complete"; data: State }
  | { type: "client_updated"; data: { bankId: string; detection: Detection } }
  | { type: "fraud_propagated"; data: { bankId: string; regime: Regime } }
  | { type: "attack_detected"; data: { bankId: string; rejected: boolean } };
