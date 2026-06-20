/**
 * In-memory fake bank node implementing the Transport surface — no network.
 * Records every update payload it receives so tests can assert on the wire
 * shape (and prove raw events are absent).
 */
import { MODEL_DIM, initWeights } from "../src/model.js";
import type { Vector } from "../src/model.js";
import type {
  EdgeModelResponse,
  EdgeUpdateRequest,
  EdgeUpdateResponse,
  Transport,
} from "../src/transport.js";

export class FakeNode implements Transport {
  version = 7;
  weights: Vector;
  received: EdgeUpdateRequest[] = [];

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
}
