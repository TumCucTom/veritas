/**
 * Injectable transport so the SDK can talk to a live bank node over fetch in
 * production, or to a fake in tests / the demo (no live node required).
 */
import type { Vector } from "./model.js";
import type { CohortAssignment } from "./secureAgg.js";

/** GET /edge/v1/model response (per PROTOCOL.md). */
export interface EdgeModelResponse {
  version: number;
  dim: number;
  weights: Vector;
}

/** POST /edge/v1/cohort request: a device asks the node to assign it to a
 * secure-aggregation cohort for the current round. The node (relay/dealer in
 * this reference; X25519 DH key-agreement coordinator in production) returns the
 * device's client id, its peers, and the per-pair shared seeds it needs to mask.
 */
export interface CohortOpenRequest {
  /** Ephemeral enrolment token — NOT a customer identifier. */
  deviceToken: string;
  /** Optional explicit round/cohort id; the node may choose one otherwise. */
  cohortId?: string;
}

/** GET/POST /edge/v1/cohort response: the device's cohort assignment. This is
 * exactly the {@link CohortAssignment} the secure-agg masking needs. */
export type CohortOpenResponse = CohortAssignment;

/** POST /edge/v1/updates request body (per PROTOCOL.md).
 *
 * In the secure-aggregation posture the `update` field carries the MASKED
 * vector (clipped + DP-noised on-device, then pairwise-masked), accompanied by
 * the cohort/round id and the device's client id within that cohort. The node
 * can only sum a cohort of these to recover the aggregate — never an
 * individual cleartext update. When no cohort is configured the SDK falls back
 * to the DP-only update (cohortId/clientId omitted) for backward compatibility.
 */
export interface EdgeUpdateRequest {
  /** Ephemeral enrolment token — NOT a customer identifier. */
  deviceToken: string;
  /** DP-protected weight delta, MASKED when cohortId is present (length = dim). */
  update: Vector;
  /** Number of local examples the update was trained on (for weighting). */
  numExamples: number;
  /** Cohort/round id this masked update belongs to (secure-agg posture). */
  cohortId?: string;
  /** This device's client id within the cohort (for mask cancellation). */
  clientId?: string;
}

export interface EdgeUpdateResponse {
  accepted: boolean;
}

/** The minimal surface the federation client needs from a node. */
export interface Transport {
  getModel(): Promise<EdgeModelResponse>;
  postUpdate(body: EdgeUpdateRequest): Promise<EdgeUpdateResponse>;
  /**
   * Open (or join) a secure-aggregation cohort for this round. Returns the
   * device's {@link CohortAssignment} (client id, peers, per-pair seeds) so the
   * device can pairwise-mask its update. Optional so legacy DP-only transports
   * still satisfy the interface; when absent, secure aggregation is unavailable
   * and the SDK falls back to the DP-only posture.
   */
  openCohort?(body: CohortOpenRequest): Promise<CohortOpenResponse>;
}

export interface FetchTransportOpts {
  nodeUrl: string;
  /** Bank-issued SDK key, sent as a bearer token. */
  key?: string;
  /** Override for environments without a global fetch. */
  fetchImpl?: typeof fetch;
}

/** Production transport: hits the bank node's /edge/v1/* endpoints. */
export class FetchTransport implements Transport {
  private readonly base: string;
  private readonly key?: string;
  private readonly doFetch: typeof fetch;

  constructor(opts: FetchTransportOpts) {
    this.base = opts.nodeUrl.replace(/\/+$/, "");
    this.key = opts.key;
    const f = opts.fetchImpl ?? globalThis.fetch;
    if (!f) {
      throw new Error(
        "No fetch available; pass fetchImpl or run on a platform with global fetch.",
      );
    }
    this.doFetch = f.bind(globalThis);
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "content-type": "application/json" };
    if (this.key) h["authorization"] = `Bearer ${this.key}`;
    return h;
  }

  async getModel(): Promise<EdgeModelResponse> {
    const res = await this.doFetch(`${this.base}/edge/v1/model`, {
      method: "GET",
      headers: this.headers(),
    });
    if (!res.ok) {
      throw new Error(`GET /edge/v1/model failed: ${res.status}`);
    }
    return (await res.json()) as EdgeModelResponse;
  }

  async postUpdate(body: EdgeUpdateRequest): Promise<EdgeUpdateResponse> {
    const res = await this.doFetch(`${this.base}/edge/v1/updates`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      throw new Error(`POST /edge/v1/updates failed: ${res.status}`);
    }
    return (await res.json()) as EdgeUpdateResponse;
  }

  /** POST /edge/v1/cohort — request a secure-aggregation cohort assignment so
   * the device can pairwise-mask its update. This is what makes secure
   * aggregation reachable through the production transport. */
  async openCohort(body: CohortOpenRequest): Promise<CohortOpenResponse> {
    const res = await this.doFetch(`${this.base}/edge/v1/cohort`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      throw new Error(`POST /edge/v1/cohort failed: ${res.status}`);
    }
    return (await res.json()) as CohortOpenResponse;
  }
}
