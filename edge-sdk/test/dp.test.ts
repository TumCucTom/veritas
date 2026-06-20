import { describe, it, expect } from "vitest";
import { clipUpdate, addNoise, privatize, l2norm, Rng, SecureRng } from "../src/index.js";
describe("differential privacy", () => {
  it("clip bounds the L2 norm to maxNorm", () => {
    const big = [10, -10, 10, -10, 10, 5, 5, 5, 5, 5, 5];
    const maxNorm = 3.0;
    expect(l2norm(big)).toBeGreaterThan(maxNorm);
    const clipped = clipUpdate(big, maxNorm);
    // allow tiny float epsilon
    expect(l2norm(clipped)).toBeLessThanOrEqual(maxNorm + 1e-9);
    expect(l2norm(clipped)).toBeCloseTo(maxNorm, 6);
  });

  it("clip is a no-op when already inside the ball", () => {
    const small = [0.1, -0.1, 0.05, 0, 0, 0, 0, 0, 0, 0, 0];
    const clipped = clipUpdate(small, 3.0);
    expect(clipped).toEqual(small);
  });

  it("noise is non-zero and perturbs the update", () => {
    const u = new Array(11).fill(0.5);
    const noised = addNoise(u, 0.2, 3.0, new Rng(1));
    let changed = 0;
    for (let i = 0; i < u.length; i++) if (noised[i] !== u[i]) changed++;
    expect(changed).toBe(u.length);
  });

  it("privatize bounds the clipped pre-noise norm and stays finite", () => {
    const big = new Array(11).fill(5);
    const out = privatize(big, 3.0, 0.1, new Rng(2));
    expect(out.length).toBe(big.length);
    expect(out.every((v) => Number.isFinite(v))).toBe(true);
    // with small sigma the privatized norm stays in a sane neighbourhood of maxNorm
    expect(l2norm(out)).toBeLessThan(3.0 * 4);
  });

  it("SecureRng (CSPRNG) draws non-deterministic noise — SPEC-DP", () => {
    // No fixed seed: two SecureRng instances must not produce the same stream.
    const u = new Array(64).fill(0);
    const a = addNoise(u, 0.5, 3.0, new SecureRng());
    const b = addNoise(u, 0.5, 3.0, new SecureRng());
    const identical = a.every((x, i) => x === b[i]);
    expect(identical).toBe(false);
    expect(a.every((x) => Number.isFinite(x))).toBe(true);
  });
});
