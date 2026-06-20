/**
 * Differential privacy on the model update before it leaves the device —
 * a faithful port of core/veritas_core/dp.py.
 *
 *   clip_update(u, maxNorm): scale down to the L2 ball of radius maxNorm.
 *   add_noise(u, sigma, maxNorm, rng): add N(0, (sigma*maxNorm)^2) per coord.
 *   privatize = add_noise(clip_update(...)).
 *
 * This is what makes "raw events never leave" hold even for the update: the
 * gradient delta is bounded and noised before transport.
 */
import type { Vector } from "./model.js";
import { l2norm } from "./model.js";
import type { RandomSource } from "./rng.js";

/** Clip an update to a maximum L2 norm (no-op if already inside the ball). */
export function clipUpdate(u: Vector, maxNorm: number): Vector {
  const nrm = l2norm(u);
  if (nrm <= maxNorm || nrm === 0) return u.slice();
  const scale = maxNorm / nrm;
  return u.map((v) => v * scale);
}

/** Add Gaussian noise with std = sigma * maxNorm to each coordinate. */
export function addNoise(
  u: Vector,
  sigma: number,
  maxNorm: number,
  rng: RandomSource,
): Vector {
  const std = sigma * maxNorm;
  return u.map((v) => v + rng.normal(0, std));
}

/** privatize(u) = add_noise(clip_update(u)).
 *
 * ``rng`` is the privacy-NOISE source — pass a {@link SecureRng} in production
 * (CSPRNG) and a seeded {@link Rng} only in tests (SPEC-DP). It must be distinct
 * from any synthetic-data generator. */
export function privatize(
  u: Vector,
  maxNorm: number,
  sigma: number,
  rng: RandomSource,
): Vector {
  return addNoise(clipUpdate(u, maxNorm), sigma, maxNorm, rng);
}

export interface DpParams {
  /** L2 clipping bound for the raw update. */
  maxNorm: number;
  /** Noise multiplier (std = sigma * maxNorm). */
  sigma: number;
}

export const DEFAULT_DP: DpParams = { maxNorm: 3.0, sigma: 0.1 };
