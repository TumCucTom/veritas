/**
 * In-memory fake bank node implementing the Transport surface — no network.
 * Records every update payload it receives so tests can assert on the wire
 * shape (and prove raw events are absent).
 */
import { MODEL_DIM, initWeights } from "../src/model.js";
import type { Vector } from "../src/model.js";
import { pairKey } from "../src/secureAgg.js";
import type {
  CohortOpenRequest,
  CohortOpenResponse,
  EdgeModelResponse,
  EdgeUpdateRequest,
  EdgeUpdateResponse,
  Transport,
} from "../src/transport.js";

export class FakeNode implements Transport {
  version = 7;
  weights: Vector;
  received: EdgeUpdateRequest[] = [];
  cohortRequests: CohortOpenRequest[] = [];
  /** Peers the node will assign this device to (relay/dealer stand-in). */
  cohortPeers: string[] = ["devB"];
  /** Per-pair seed (64-hex) the node deals for the cohort. */
  cohortSeed = "abcd1234".repeat(8);

  constructor(weights?: Vector) {
    this.weights = weights ?? initWeights(MODEL_DIM - 1);
  }

  async getModel(): Promise<EdgeModelResponse> {
    return { version: this.version, dim: MODEL_DIM, weights: this.weights.slice() };
  }

  async postUpdate(body: EdgeUpdateRequest): Promise<EdgeUpdateResponse> {
    // Deep-copy so later mutation can't retroactively change what we captured.
    this.received.push(JSON.parse(JSON.stringify(body)) as EdgeUpdateRequest);
    return { accepted: true };
  }

  /** Assign the requesting device to a 2-party cohort with one shared seed —
   * the trusted-dealer stand-in for X25519 DH key agreement. */
  async openCohort(body: CohortOpenRequest): Promise<CohortOpenResponse> {
    this.cohortRequests.push(JSON.parse(JSON.stringify(body)) as CohortOpenRequest);
    const clientId = "devA";
    const seeds: Record<string, string> = {};
    for (const peer of this.cohortPeers) {
      seeds[pairKey(clientId, peer)] = this.cohortSeed;
    }
    return {
      cohortId: body.cohortId ?? "round-1",
      clientId,
      peerIds: this.cohortPeers.slice(),
      seeds,
    };
  }
}
