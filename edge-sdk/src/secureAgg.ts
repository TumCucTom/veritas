/**
 * Bonawitz-style additive pairwise-masking — device side.
 *
 * This is the TypeScript counterpart of core/veritas_core/secure_agg.py. A
 * device masks its (already clipped + DP-noised) update so that the bank node
 * only ever sees a masked vector; when the node sums a cohort of masked
 * vectors, every pairwise mask cancels and it recovers ONLY the aggregate sum.
 * No individual device update is ever visible to the node in the clear.
 *
 * For every pair of clients (i, j) there is a shared secret seed s_ij. Both
 * clients derive the SAME pseudo-random mask m_ij = PRG(s_ij) from it. The
 * lower id ADDS it, the higher id SUBTRACTS it, so the pair's masks cancel in
 * the sum. The PRG here MUST byte-for-byte match the Python
 * `derive_pairwise_mask` so a TS-masked update and a Python `secure_sum`
 * interoperate — but in this single-language reference the masking and summing
 * both happen in their own runtime, and the algebra (sign by id ordering +
 * deterministic seed→mask) is what matters and is identical.
 *
 * PRODUCTION: s_ij is established by an authenticated X25519 Diffie-Hellman
 * exchange between device public keys, so the node (which only relays public
 * keys) never learns any seed and is structurally unable to unmask an
 * individual. THIS REFERENCE: the node hands out the per-pair seeds it dealt
 * (trusted-dealer stand-in), so the masking algebra can be exercised
 * end-to-end without a crypto dependency.
 */
import type { Vector } from "./model.js";
import { Rng } from "./rng.js";

/** Mask magnitude — must dominate the real update so a single masked vector
 * leaks nothing. Matches core's _MASK_SCALE. */
export const MASK_SCALE = 1.0e6;

/** A seed table maps a canonical "lowId|highId" pair key to a numeric seed. */
export type SeedTable = Record<string, number>;

/** Canonical pair key (lower id first), matching core's _canon. */
export function pairKey(a: string, b: string): string {
  return a < b ? `${a}|${b}` : `${b}|${a}`;
}

/**
 * PRG: expand a numeric shared seed into a deterministic mask vector. Both
 * clients in a pair derive the identical vector from the identical seed.
 * Uses the SDK's deterministic Rng so the same seed always yields the same
 * mask within this runtime.
 */
export function derivePairwiseMask(seed: number, dim: number): Vector {
  if (dim <= 0) throw new Error("dim must be positive");
  const rng = new Rng(seed >>> 0);
  const out = new Array<number>(dim);
  for (let i = 0; i < dim; i++) {
    // uniform in [-MASK_SCALE, MASK_SCALE)
    out[i] = (rng.next() * 2 - 1) * MASK_SCALE;
  }
  return out;
}

/**
 * Apply this client's pairwise masks to its raw (clipped + DP-noised) update.
 * For each peer p: ADD m_{client,p} if client < p else SUBTRACT it. Returns the
 * masked vector the device sends; the unmasked update never leaves the device.
 */
export function maskUpdate(
  update: Vector,
  clientId: string,
  peerIds: readonly string[],
  seeds: SeedTable,
): Vector {
  const masked = update.slice();
  const dim = masked.length;
  for (const p of peerIds) {
    if (p === clientId) continue;
    const key = pairKey(clientId, p);
    const seed = seeds[key];
    if (seed === undefined) {
      throw new Error(`no shared seed for pair ${key}`);
    }
    const m = derivePairwiseMask(seed, dim);
    const sign = clientId < p ? 1 : -1;
    for (let i = 0; i < dim; i++) masked[i] = masked[i]! + sign * m[i]!;
  }
  return masked;
}

/** Cohort assignment a device receives from the node (relay/dealer in the
 * reference; X25519 DH public-key exchange in production). */
export interface CohortAssignment {
  /** Round / cohort identifier this masked update belongs to. */
  cohortId: string;
  /** This device's client id within the cohort. */
  clientId: string;
  /** The other client ids in the cohort (peers to mask against). */
  peerIds: string[];
  /** Per-pair shared seeds (only the pairs that include clientId are needed). */
  seeds: SeedTable;
}
