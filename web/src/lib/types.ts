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
export interface State {
  round: number;
  running: boolean;
  banks: Bank[];
  counters: Counters;
  campaignActive: boolean;
  attackActive: boolean;
  customerRecordsTransmitted: number;
}
export interface Provenance {
  round: number;
  contributors: string[];
  rejected: string[];
  globalRecall: number;
}
export type Regime = "federated" | "siloed";
export type VeritasEvent =
  | { type: "round_complete"; data: State }
  | { type: "client_updated"; data: { bankId: string; detection: Detection } }
  | { type: "fraud_propagated"; data: { bankId: string; regime: Regime } }
  | { type: "attack_detected"; data: { bankId: string; rejected: boolean } };
