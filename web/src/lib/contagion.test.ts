import { describe, expect, it } from "vitest";
import {
  LOGICAL_CUSTOMERS,
  VISIBLE_POINTS,
  computeContagionFrame,
} from "./contagion";

describe("computeContagionFrame", () => {
  it("models a million logical customers with a bounded visible sample", () => {
    const frame = computeContagionFrame({
      regime: "siloed",
      round: 3,
      campaignActive: true,
      detection: 0.42,
    });

    expect(LOGICAL_CUSTOMERS).toBe(1_000_000);
    expect(frame.logicalCustomers).toBe(LOGICAL_CUSTOMERS);
    expect(frame.points.length).toBe(VISIBLE_POINTS);
    expect(frame.points.length).toBeLessThan(LOGICAL_CUSTOMERS);
    expect(frame.scalePerPoint).toBe(LOGICAL_CUSTOMERS / VISIBLE_POINTS);
  });

  it("is deterministic for the same state", () => {
    const a = computeContagionFrame({
      regime: "federated",
      round: 5,
      campaignActive: true,
      detection: 0.91,
    });
    const b = computeContagionFrame({
      regime: "federated",
      round: 5,
      campaignActive: true,
      detection: 0.91,
    });

    expect(a.totals).toEqual(b.totals);
    expect(a.frontier).toEqual(b.frontier);
    expect(a.points.slice(0, 20)).toEqual(b.points.slice(0, 20));
  });

  it("lets siloed fraud exposure spread as rounds advance", () => {
    const early = computeContagionFrame({
      regime: "siloed",
      round: 1,
      campaignActive: true,
      detection: 0.18,
    });
    const later = computeContagionFrame({
      regime: "siloed",
      round: 6,
      campaignActive: true,
      detection: 0.38,
    });

    expect(later.totals.exposed).toBeGreaterThan(early.totals.exposed);
    expect(later.frontier.fraudRadius).toBeGreaterThan(early.frontier.fraudRadius);
  });

  it("makes federated protection overtake fraud as detection rises", () => {
    const siloed = computeContagionFrame({
      regime: "siloed",
      round: 6,
      campaignActive: true,
      detection: 0.38,
    });
    const federated = computeContagionFrame({
      regime: "federated",
      round: 6,
      campaignActive: true,
      detection: 0.96,
    });

    expect(federated.totals.protected).toBeGreaterThan(siloed.totals.protected);
    expect(federated.totals.exposed).toBeLessThan(siloed.totals.exposed);
    expect(federated.frontier.protectionRadius).toBeGreaterThan(federated.frontier.fraudRadius);
  });
});
