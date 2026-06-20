/*
 * TypeScript types for the Tier 2 Control Plane API (the per-tenant production
 * surface), mirroring protocol/PROTOCOL.md §"Tier 2 — Control Plane API".
 *
 * These are intentionally separate from ./types.ts (the single-node demo
 * contract) so the two data sources — the live node race and the federation /
 * governance / provenance control plane — stay clearly delineated.
 */

/** GET /v1/tenants/{tid}/state */
export interface TenantLiftRegime {
  recall: number;
  victims: number;
  lostGbp: number;
}
export interface TenantLift {
  federated: TenantLiftRegime;
  siloed: TenantLiftRegime;
  fraudPreventedGbp: number;
}
export interface TenantNode {
  status: string;
  lastSync: string;
  attestation: "verified" | "unknown";
}
export interface TenantState {
  round: number;
  modelVersion: number;
  /** Architecturally pinned to 0 — the sovereignty proof. */
  customerRecordsTransmitted: number;
  node: TenantNode;
  lift: TenantLift;
  /**
   * Optional edge-fleet coverage. The production control plane may not yet
   * expose Tier-0 metrics; when present we render live, otherwise the edge
   * panel falls back to a clearly-labelled representative view.
   */
  edgeFleet?: EdgeFleet;
  gnnBenchmark?: import("./types").GnnBenchmark;
}

/** Tier-0 edge fleet coverage (optional on tenant state). */
export interface EdgeFleet {
  devicesEnrolled?: number;
  onLatestModelPct?: number;
  modelVersion?: number;
  scamInProgressAlerts24h?: number;
  appVersions?: { version: string; share: number }[];
}

/** GET /v1/tenants/{tid}/provenance */
export interface TenantProvenance {
  round: number;
  contributors: string[];
  rejected: string[];
  globalRecall: number;
}

/** GET /v1/transparency/root — the signed Merkle root (tamper-evidence). */
export interface TransparencyRoot {
  size: number;
  rootHash: string;
  signaturePem: string;
}

/** GET /v1/transparency — append-only log entries. */
export interface TransparencyEntry {
  seq: number;
  type:
    | "round_aggregated"
    | "model_promoted"
    | "model_rolledback"
    | "member_enrolled"
    | string;
  round?: number;
  data: Record<string, unknown>;
  leafHash: string;
  timestamp: string;
}

/** GET /v1/models/registry */
export type ModelStatus = "canary" | "promoted" | "rolledback";
export interface ModelRegistryEntry {
  version: number;
  status: ModelStatus;
  metrics: Record<string, number> & { recall?: number };
  createdAt: string;
}

/** GET /v1/events?tenantId= (SSE) named events. */
export type ControlPlaneEvent =
  | { type: "round_complete"; data: unknown }
  | { type: "model_promoted"; data: unknown }
  | { type: "attack_detected"; data: unknown }
  | { type: "member_enrolled"; data: unknown };

export type ControlPlaneEventName = ControlPlaneEvent["type"];
