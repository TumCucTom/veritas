import { describe, it, expect } from "vitest";
import { Veritas, MODEL_DIM } from "../src/index.js";
import { FakeNode } from "./fakeNode.js";

const FRAUD = {
  payeeId: "acct-new-999",
  amount: 9000,
  isNewPayee: true,
  remoteAccessAppActive: true,
  inboundCallActive: true,
  sessionAnomaly: 0.9,
};

const BENIGN = {
  payeeId: "landlord-rent",
  amount: 750,
  isNewPayee: false,
  sessionAnomaly: 0.05,
  accountAgeDays: 2000,
};

describe("observePayment", () => {
  it("returns high risk for obvious fraud, low for benign", () => {
    const node = new FakeNode();
    const v = Veritas.start({ transport: node, seed: 7, seedEvents: 400 });
    v.trainLocalModel();

    const fraud = v.observePayment(FRAUD);
    const benign = v.observePayment(BENIGN);

    expect(fraud.risk).toBeGreaterThan(0.8);
    expect(fraud.risk).toBeGreaterThan(benign.risk);
    expect(benign.risk).toBeLessThan(0.5);
    expect(fraud.reason).toMatch(/scam in progress|coercion|risk/i);
    expect(fraud.action).toBe("hold");
  });
});

describe("syncModel", () => {
  it("adopts the bank edge model and version", async () => {
    const node = new FakeNode();
    node.version = 42;
    const v = Veritas.start({ transport: node, seed: 7, seedEvents: 0 });
    const res = await v.syncModel();
    expect(res.version).toBe(42);
    expect(res.dim).toBe(MODEL_DIM);
    expect(v.version).toBe(42);
  });
});

describe("contributeUpdate", () => {
  it("sends a DP weight delta to the node and NEVER raw events", async () => {
    const node = new FakeNode();
    const v = Veritas.start({ transport: node, seed: 7, seedEvents: 400 });

    // observe a couple of payments so there's recent on-device activity
    v.observePayment(FRAUD);
    v.observePayment(BENIGN);

    const result = await v.contributeUpdate();
    expect(result.sent).toBe(true);
    expect(node.received.length).toBe(1);

    const payload = node.received[0]!;

    // exact wire shape per PROTOCOL.md
    expect(Object.keys(payload).sort()).toEqual(
      ["deviceToken", "numExamples", "update"].sort(),
    );
    expect(payload.update.length).toBe(MODEL_DIM);
    expect(payload.numExamples).toBeGreaterThan(0);
    expect(typeof payload.deviceToken).toBe("string");

    // it is a DELTA, DP-noised -> not all zeros, all finite
    expect(payload.update.some((x) => x !== 0)).toBe(true);
    expect(payload.update.every((x) => Number.isFinite(x))).toBe(true);

    // PRIVACY: no raw events, features, labels, payeeId, amount anywhere
    const json = JSON.stringify(payload);
    expect(json).not.toMatch(/features|observedAt|payeeId|amount|label|events/i);
    expect((payload as Record<string, unknown>).features).toBeUndefined();
    expect((payload as Record<string, unknown>).events).toBeUndefined();
    // deviceToken must not be a customer identifier
    expect(payload.deviceToken).not.toContain(FRAUD.payeeId);
  });

  it("DP-noised update differs from the raw delta (privacy noise applied)", async () => {
    const node = new FakeNode();
    const v = Veritas.start({
      transport: node,
      seed: 7,
      seedEvents: 400,
      dp: { maxNorm: 3.0, sigma: 0.1 },
    });
    const before = v.getWeights();
    await v.contributeUpdate();
    const after = v.getWeights();
    const rawDelta = after.map((w, i) => w - before[i]!);
    const sent = node.received[0]!.update;
    // sent != rawDelta because of clipping/noise
    const identical = rawDelta.every((d, i) => d === sent[i]);
    expect(identical).toBe(false);
  });
});
