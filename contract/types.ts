// TypeScript mirror of the Veritas API contract.
//
// SOURCE OF TRUTH: contract/schema.py (pydantic). This file mirrors that schema
// for TypeScript consumers; web/src/lib/types.ts mirrors this file in turn. Keep
// field names camelCase so the Python and TS shapes are identical over the wire.
// Change schema.py first, then update this file to match.

export interface Detection { federated: number; siloed: number; }
export interface Bank { id: string; name: string; customers: number; detection: Detection; poisoned: boolean; }
export interface RegimeCounters { fraudPreventedGbp: number; timeToDetectHours: number; victims: number; lostGbp: number; }
export interface Counters { federated: RegimeCounters; siloed: RegimeCounters; }
export interface State { round: number; running: boolean; banks: Bank[]; counters: Counters; campaignActive: boolean; attackActive: boolean; customerRecordsTransmitted: number; }
export interface Provenance { round: number; contributors: string[]; rejected: string[]; globalRecall: number; }
export type Regime = "federated" | "siloed";

// POST /predict request/response. Mirrors node/node/server/app.py: `confidence`
// is a probability in the closed interval [0, 1]. `transaction` is the
// whitelisted numeric feature map; `text` is optional free-text for triage.
export interface PredictRequest { transaction: Record<string, unknown>; text?: string; }
export interface PredictResponse { label: string; confidence: number; indicators: string[]; explanation?: string; }

export type VeritasEvent =
  | { type: "round_complete"; data: State }
  | { type: "client_updated"; data: { bankId: string; detection: Detection } }
  | { type: "fraud_propagated"; data: { bankId: string; regime: Regime } }
  | { type: "attack_detected"; data: { bankId: string; rejected: boolean } };
