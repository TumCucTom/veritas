/**
 * Injectable transport so the SDK can talk to a live bank node over fetch in
 * production, or to a fake in tests / the demo (no live node required).
 */
import type { Vector } from "./model.js";

/** GET /edge/v1/model response (per PROTOCOL.md). */
export interface EdgeModelResponse {
  version: number;
  dim: number;
  weights: Vector;
}

/** POST /edge/v1/updates request body (per PROTOCOL.md). */
export interface EdgeUpdateRequest {
  /** Ephemeral enrolment token — NOT a customer identifier. */
  deviceToken: string;
  /** DP-protected weight delta (length = dim). */
  update: Vector;
  /** Number of local examples the update was trained on (for weighting). */
  numExamples: number;
}

export interface EdgeUpdateResponse {
  accepted: boolean;
}

/** The minimal surface the federation client needs from a node. */
export interface Transport {
  getModel(): Promise<EdgeModelResponse>;
  postUpdate(body: EdgeUpdateRequest): Promise<EdgeUpdateResponse>;
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
}
