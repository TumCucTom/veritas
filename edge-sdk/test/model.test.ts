import { describe, it, expect } from "vitest";
import {
  FEATURE_DIM,
  MODEL_DIM,
  initWeights,
  trainLocal,
  recall,
  generateSyntheticEvents,
} from "../src/index.js";

function toXY(events: ReturnType<typeof generateSyntheticEvents>) {
  return { X: events.map((e) => e.features), y: events.map((e) => e.label) };
}

describe("on-device logistic model", () => {
  it("has dim = FEATURE_DIM + 1", () => {
    expect(initWeights(FEATURE_DIM).length).toBe(MODEL_DIM);
    expect(MODEL_DIM).toBe(11);
  });

  it("training improves recall on synthetic data", () => {
    const events = generateSyntheticEvents(600, 0.15, 42);
    const { X, y } = toXY(events);

    const w0 = initWeights(FEATURE_DIM);
    const recallBefore = recall(w0, X, y);

    const w1 = trainLocal(w0, X, y, { epochs: 20 });
    const recallAfter = recall(w1, X, y);

    expect(recallAfter).toBeGreaterThan(recallBefore);
    // class-reweighted logistic should recover most positives
    expect(recallAfter).toBeGreaterThan(0.7);
  });
});
