/**
 * Veritas Tier 0 Edge SDK — public surface.
 */
export { Veritas } from "./veritas.js";
export type { VeritasConfig, ContributeResult } from "./veritas.js";
export type { PaymentObservation, DeviceEvent } from "./events.js";
export {
  generateSyntheticEvents,
  paymentToFeatures,
  EventBuffer,
} from "./events.js";
export type { RiskVerdict } from "./scam.js";
export { assessScam } from "./scam.js";
export {
  FEATURE_DIM,
  MODEL_DIM,
  initWeights,
  predictProba,
  predictProbaOne,
  trainLocal,
  recall,
  subtract,
  l2norm,
} from "./model.js";
export type { Vector, Matrix } from "./model.js";
export { clipUpdate, addNoise, privatize, DEFAULT_DP } from "./dp.js";
export type { DpParams } from "./dp.js";
export { FetchTransport } from "./transport.js";
export type {
  Transport,
  EdgeModelResponse,
  EdgeUpdateRequest,
  EdgeUpdateResponse,
  FetchTransportOpts,
  CohortOpenRequest,
  CohortOpenResponse,
} from "./transport.js";
export { Rng, SecureRng } from "./rng.js";
export type { RandomSource } from "./rng.js";
export {
  maskUpdate,
  derivePairwiseMask,
  prgFloats,
  pairKey,
  seedToHex,
  hexToSeed,
  MASK_SCALE,
  SEED_BYTES,
} from "./secureAgg.js";
export type { SeedTable, CohortAssignment } from "./secureAgg.js";
