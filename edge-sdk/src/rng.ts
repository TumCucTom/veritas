/**
 * Small deterministic PRNG (mulberry32) + Gaussian sampler (Box–Muller).
 *
 * Mirrors the role of numpy's `default_rng` in the Python core: a seedable
 * source so synthetic DATA and tests are reproducible. No native deps — runs
 * identically in browser and node.
 *
 * IMPORTANT (SPEC-DP): this deterministic Rng is for synthetic-data generation
 * and for tests that pass an explicit seed. Production DP privacy NOISE must NOT
 * be drawn from a constant-seeded PRNG — use `SecureRng` (crypto.getRandomValues)
 * for that, and keep it SEPARATE from the data Rng.
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

/**
 * A source of uniform/normal samples. Both the deterministic {@link Rng} and the
 * cryptographically-secure {@link SecureRng} satisfy it, so DP noise can be
 * drawn from either (secure by default in production, seeded only in tests).
 */
export interface RandomSource {
  next(): number;
  normal(mean?: number, std?: number): number;
}

/**
 * Cryptographically-secure RNG for DP privacy NOISE (SPEC-DP).
 *
 * Draws uniforms from `crypto.getRandomValues` (a CSPRNG available in browsers
 * and modern Node), so the noise is never derived from a constant seed. Gaussian
 * samples use the same Box–Muller transform as {@link Rng} for an identical
 * noise distribution — only the entropy source differs.
 */
export class SecureRng implements RandomSource {
  private readonly cryptoObj: Crypto;

  constructor(cryptoObj?: Crypto) {
    const c = cryptoObj ?? (globalThis.crypto as Crypto | undefined);
    if (!c || typeof c.getRandomValues !== "function") {
      throw new Error(
        "No secure crypto.getRandomValues available; cannot draw DP noise.",
      );
    }
    this.cryptoObj = c;
  }

  /** Uniform float in [0, 1) from 32 bits of CSPRNG output. */
  next(): number {
    const buf = new Uint32Array(1);
    this.cryptoObj.getRandomValues(buf);
    return buf[0]! / 4294967296;
  }

  /** Standard normal sample N(0, 1) via Box–Muller. */
  normal(mean = 0, std = 1): number {
    let u1 = 0;
    while (u1 === 0) u1 = this.next();
    const u2 = this.next();
    const mag = Math.sqrt(-2.0 * Math.log(u1));
    return mean + std * mag * Math.cos(2.0 * Math.PI * u2);
  }
}
