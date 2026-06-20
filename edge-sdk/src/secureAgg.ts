/**
 * Bonawitz-style additive pairwise-masking — device side.
 *
 * This is the TypeScript counterpart of core/veritas_core/secure_agg.py. A
 * device masks its (already clipped + DP-noised) update so that the bank node
 * only ever sees a masked vector; when the node sums a cohort of masked
 * vectors, every pairwise mask cancels and it recovers ONLY the aggregate sum.
 * No individual device update is ever visible to the node in the clear.
 *
 * For every pair of clients (i, j) there is a shared secret seed s_ij (32 raw
 * bytes, carried on the wire as 64-char lowercase hex). Both clients derive the
 * SAME pseudo-random mask m_ij = PRG(s_ij) from it. The lower id ADDS it, the
 * higher id SUBTRACTS it, so the pair's masks cancel in the sum.
 *
 * The PRG here is byte-for-byte IDENTICAL to the Python canonical
 * `veritas_core.secure_agg.prg_floats` (HMAC-SHA256 in counter mode, four
 * little-endian uint64 chunks per 32-byte block, each divided by 2^64). A
 * TS-masked update and a Python `secure_sum` therefore genuinely interoperate:
 * the masks cancel because the underlying float bytes are the same in both
 * languages. A pinned golden test vector (32 bytes of 0x01) is asserted in BOTH
 * the Python and TypeScript test suites so any divergence is caught.
 *
 * PRODUCTION: s_ij is established by an authenticated X25519 Diffie-Hellman
 * exchange between device public keys, so the node (which only relays public
 * keys) never learns any seed and is structurally unable to unmask an
 * individual. THIS REFERENCE: the node hands out the per-pair seeds it dealt
 * (trusted-dealer stand-in), so the masking algebra can be exercised
 * end-to-end without a crypto dependency.
 */
import { createHmac } from "node:crypto";
import type { Vector } from "./model.js";

/** Mask magnitude — must dominate the real update so a single masked vector
 * leaks nothing. Matches core's _MASK_SCALE. */
export const MASK_SCALE = 1.0e6;

/** Canonical seed length in bytes (256-bit HMAC-SHA256 key). */
export const SEED_BYTES = 32;

/** A seed table maps a canonical "lowId|highId" pair key to a 64-hex seed. */
export type SeedTable = Record<string, string>;

/** Canonical pair key (lower id first), matching core's _canon. */
export function pairKey(a: string, b: string): string {
  return a < b ? `${a}|${b}` : `${b}|${a}`;
}

/** 32-byte seed -> 64-char lowercase hex (wire form). Matches core seed_to_hex. */
export function seedToHex(seed: Uint8Array): string {
  if (seed.length !== SEED_BYTES) {
    throw new Error(`seed must be ${SEED_BYTES} bytes`);
  }
  return Buffer.from(seed).toString("hex");
}

/** 64-char hex -> 32 raw bytes. Matches core hex_to_seed. */
export function hexToSeed(hex: string): Buffer {
  const b = Buffer.from(hex, "hex");
  if (b.length !== SEED_BYTES) {
    throw new Error(`seed hex must decode to ${SEED_BYTES} bytes`);
  }
  return b;
}

/**
 * Canonical cross-language PRG: expand a 32-byte seed into `n` uniform float64s
 * in [0, 1). HMAC-SHA256 counter mode, four little-endian uint64 chunks per
 * 32-byte block divided by 2^64. Byte-identical to Python `prg_floats`.
 */
export function prgFloats(seed: Buffer, n: number): Vector {
  if (n < 0) throw new Error("n must be non-negative");
  const out = new Array<number>(n);
  let filled = 0;
  let counter = 0;
  const TWO_POW_64 = 2 ** 64;
  while (filled < n) {
    const msg = Buffer.alloc(8);
    // counter as 8-byte big-endian (counter stays well under 2^32 here).
    msg.writeUInt32BE(counter >>> 0, 4);
    const block = createHmac("sha256", seed).update(msg).digest(); // 32 bytes
    for (let j = 0; j < 4 && filled < n; j++) {
      // little-endian uint64 from 8 bytes -> divide by 2^64.
      const lo = block.readUInt32LE(j * 8);
      const hi = block.readUInt32LE(j * 8 + 4);
      const val = hi * 0x1_0000_0000 + lo; // uint64 as a JS double
      out[filled++] = val / TWO_POW_64;
    }
    counter++;
  }
  return out;
}

/**
 * PRG: expand a hex shared seed into a deterministic mask vector. Both clients
 * in a pair derive the identical vector from the identical 64-hex seed, and the
 * Python core derives the SAME vector from the same seed bytes (byte-for-byte
 * interop — see prgFloats).
 */
export function derivePairwiseMask(seedHex: string, dim: number): Vector {
  if (dim <= 0) throw new Error("dim must be positive");
  const u = prgFloats(hexToSeed(seedHex), dim);
  const out = new Array<number>(dim);
  for (let i = 0; i < dim; i++) {
    // uniform [0,1) -> zero-centred, large magnitude (matches core).
    out[i] = (u[i]! * 2 - 1) * MASK_SCALE;
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
