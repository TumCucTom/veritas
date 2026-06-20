/**
 * Small deterministic PRNG (mulberry32) + Gaussian sampler (Box–Muller).
 *
 * Mirrors the role of numpy's `default_rng` in the Python core: a seedable
 * source so synthetic data, DP noise, and tests are reproducible. No native
 * deps — runs identically in browser and node.
 */
export class Rng {
  private state: number;

  constructor(seed = 0x9e3779b9) {
    // Avoid a zero state (mulberry32 degenerates at 0).
    this.state = seed >>> 0 || 0x9e3779b9;
  }

  /** Uniform float in [0, 1). */
  next(): number {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Standard normal sample N(0, 1) via Box–Muller. */
  normal(mean = 0, std = 1): number {
    let u1 = 0;
    let u2 = 0;
    // Guard against log(0).
    while (u1 === 0) u1 = this.next();
    u2 = this.next();
    const mag = Math.sqrt(-2.0 * Math.log(u1));
    return mean + std * mag * Math.cos(2.0 * Math.PI * u2);
  }
}
