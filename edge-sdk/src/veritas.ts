/**
 * Veritas Tier 0 Edge SDK — public entry point.
 *
 * Ergonomics mirror an analytics SDK (spec §3.3):
 *   const veritas = Veritas.start({ nodeUrl, key });
 *   const { risk, reason } = veritas.observePayment({ payeeId, amount, isNewPayee });
 *   await veritas.syncModel();
 *   await veritas.contributeUpdate();
 *
 * Privacy invariant: raw events live only in the on-device buffer. The only
 * thing that ever leaves is a clipped + Gaussian-noised weight DELTA.
 */
import {
  FEATURE_DIM,
  MODEL_DIM,
  initWeights,
  predictProbaOne,
  trainLocal,
  recall,
  subtract,
} from "./model.js";
import type { Vector } from "./model.js";
import { DEFAULT_DP, privatize } from "./dp.js";
import type { DpParams } from "./dp.js";
import {
  EventBuffer,
  generateSyntheticEvents,
  paymentToFeatures,
  toTrainingSet,
} from "./events.js";
import type { DeviceEvent, PaymentObservation } from "./events.js";
import { assessScam } from "./scam.js";
import type { RiskVerdict } from "./scam.js";
import { FetchTransport } from "./transport.js";
import type {
  EdgeUpdateRequest,
  EdgeUpdateResponse,
  Transport,
} from "./transport.js";
import { maskUpdate } from "./secureAgg.js";
import type { CohortAssignment } from "./secureAgg.js";
import { Rng, SecureRng } from "./rng.js";
import type { RandomSource } from "./rng.js";

export interface VeritasConfig {
  /** Base URL of the bank's Veritas Node (Tier 1). */
  nodeUrl?: string;
  /** Bank-issued SDK key (bearer token). */
  key?: string;
  /** Ephemeral enrolment token sent with updates (NOT a customer id). */
  deviceToken?: string;
  /** Inject a transport for tests / demos; defaults to FetchTransport. */
  transport?: Transport;
  /** DP clipping + noise parameters. */
  dp?: DpParams;
  /** Seed for synthetic data + DP noise (reproducible tests). */
  seed?: number;
  /**
   * Seed the local buffer with N synthetic events at start (simulates a device
   * that has already observed traffic). Set 0 to start empty.
   */
  seedEvents?: number;
  fraudRate?: number;
}

export interface ContributeResult {
  sent: boolean;
  numExamples: number;
  /** L2 norm of the DP-protected update actually transmitted. */
  updateNorm: number;
  localRecall: number;
  response?: EdgeUpdateResponse;
}

export class Veritas {
  private weights: Vector;
  private modelVersion = 0;
  private readonly buffer: EventBuffer;
  private readonly transport?: Transport;
  private readonly dp: DpParams;
  /** Privacy-NOISE source (SPEC-DP). Secure CSPRNG by default; a seeded,
   * deterministic Rng ONLY when the caller passed an explicit `seed` (tests).
   * Kept SEPARATE from the synthetic-data generator. */
  private readonly noiseRng: RandomSource;
  private readonly deviceToken: string;

  private constructor(cfg: VeritasConfig) {
    this.weights = initWeights(FEATURE_DIM); // dim = FEATURE_DIM + 1
    this.dp = cfg.dp ?? DEFAULT_DP;
    // SPEC-DP: deterministic noise ONLY behind an explicit test seed. With no
    // seed, draw privacy noise from a cryptographically secure RNG. The seeded
    // noise stream is offset from the data seed so the two generators differ.
    this.noiseRng =
      cfg.seed !== undefined ? new Rng((cfg.seed ^ 0x5eed0015) >>> 0) : new SecureRng();
    this.deviceToken = cfg.deviceToken ?? `dev-${(cfg.seed ?? 1234).toString(36)}`;
    this.buffer = new EventBuffer();

    if (cfg.transport) {
      this.transport = cfg.transport;
    } else if (cfg.nodeUrl) {
      this.transport = new FetchTransport({
        nodeUrl: cfg.nodeUrl,
        key: cfg.key,
      });
    }

    const seedN = cfg.seedEvents ?? 400;
    if (seedN > 0) {
      this.buffer.addMany(
        generateSyntheticEvents(seedN, cfg.fraudRate ?? 0.12, cfg.seed ?? 1234),
      );
    }
  }

  /** One-line init (spec §3.3). */
  static start(cfg: VeritasConfig = {}): Veritas {
    return new Veritas(cfg);
  }

  /** Current on-device model dimension (FEATURE_DIM + 1). */
  get dim(): number {
    return MODEL_DIM;
  }

  /** Current edge model version pulled from the node (0 if never synced). */
  get version(): number {
    return this.modelVersion;
  }

  /** Read-only snapshot of the current weights (for tests / debugging). */
  getWeights(): Vector {
    return this.weights.slice();
  }

  /** Number of raw events currently buffered on-device. */
  bufferSize(): number {
    return this.buffer.size();
  }

  /**
   * Score a payment locally, on-device, using the current model + the
   * scam-in-progress rule layer. Adds the observation to the local buffer.
   * Returns { risk, reason } (plus action + indicators).
   *
   * The host app may pass a known label to teach the device (e.g. the user
   * confirmed it was a scam) — used only locally for training.
   */
  observePayment(
    p: PaymentObservation,
    label?: 0 | 1,
  ): RiskVerdict {
    const features = paymentToFeatures(p);
    const modelScore = predictProbaOne(this.weights, features);
    const verdict = assessScam(modelScore, p);

    // Buffer the raw event on-device. If no explicit label, weak-label from the
    // local verdict (held/hold => suspected fraud) so the device keeps learning.
    const inferredLabel: 0 | 1 =
      label !== undefined ? label : verdict.action === "hold" ? 1 : 0;
    const event: DeviceEvent = {
      features,
      label: inferredLabel,
      observedAt: Date.now(),
    };
    this.buffer.add(event);
    return verdict;
  }

  /** Train the model on the local buffer (raw events never leave). */
  trainLocalModel(epochs = 12): { recall: number; numExamples: number } {
    const { X, y } = toTrainingSet(this.buffer.all());
    if (y.length === 0) return { recall: 1.0, numExamples: 0 };
    this.weights = trainLocal(this.weights, X, y, { epochs });
    return { recall: recall(this.weights, X, y), numExamples: y.length };
  }

  /** GET the bank edge model and adopt it as the local base. */
  async syncModel(): Promise<{ version: number; dim: number }> {
    if (!this.transport) {
      throw new Error("No transport configured: pass nodeUrl or transport.");
    }
    const m = await this.transport.getModel();
    if (m.dim !== MODEL_DIM || m.weights.length !== MODEL_DIM) {
      throw new Error(
        `Edge model dim ${m.dim}/${m.weights.length} != expected ${MODEL_DIM}.`,
      );
    }
    this.weights = m.weights.slice();
    this.modelVersion = m.version;
    return { version: m.version, dim: m.dim };
  }

  /**
   * Train locally on the device buffer, DP-protect the weight DELTA (clip +
   * Gaussian noise), optionally MASK it for secure aggregation, and POST it to
   * the node. RAW EVENTS ARE NEVER INCLUDED in the payload — only the noised
   * (and, under secure-agg, masked) delta + a count.
   *
   * When `opts.cohort` is provided the device performs Bonawitz-style pairwise
   * masking on top of its DP update, so the bank node sees only the masked
   * vector and can only ever recover the cohort SUM — never this device's
   * individual update. Without a cohort, the DP-only update is sent (legacy
   * posture, still privacy-preserving via DP but visible per-device).
   */
  async contributeUpdate(
    opts: {
      epochs?: number;
      cohort?: CohortAssignment;
      /** Fetch a secure-aggregation cohort from the node transport before
       * contributing (requires a transport implementing `openCohort`). Ignored
       * when an explicit `cohort` is supplied. */
      useCohort?: boolean;
      /** Optional cohort/round id to request when `useCohort` is set. */
      cohortId?: string;
    } = {},
  ): Promise<ContributeResult> {
    if (!this.transport) {
      throw new Error("No transport configured: pass nodeUrl or transport.");
    }

    // Resolve the cohort: explicit assignment wins; otherwise, if requested,
    // open one through the real transport so secure aggregation is reachable in
    // production (not only via a hand-built CohortAssignment).
    let cohort = opts.cohort;
    if (!cohort && opts.useCohort) {
      if (!this.transport.openCohort) {
        throw new Error(
          "Transport does not support openCohort; cannot open a secure-agg cohort.",
        );
      }
      cohort = await this.transport.openCohort({
        deviceToken: this.deviceToken,
        cohortId: opts.cohortId,
      });
    }

    const base = this.weights.slice();
    const { X, y } = toTrainingSet(this.buffer.all());
    if (y.length === 0) {
      return { sent: false, numExamples: 0, updateNorm: 0, localRecall: 1.0 };
    }

    const trained = trainLocal(base, X, y, { epochs: opts.epochs ?? 12 });
    this.weights = trained; // keep the improved local model

    const rawDelta = subtract(trained, base);
    // DP happens CLIENT-SIDE: clip + Gaussian noise the delta.
    const dpUpdate = privatize(rawDelta, this.dp.maxNorm, this.dp.sigma, this.noiseRng);

    // Secure aggregation: mask the DP update so the node never sees it in clear.
    const body: EdgeUpdateRequest = cohort
      ? {
          deviceToken: this.deviceToken,
          update: maskUpdate(
            dpUpdate,
            cohort.clientId,
            cohort.peerIds,
            cohort.seeds,
          ),
          numExamples: y.length,
          cohortId: cohort.cohortId,
          clientId: cohort.clientId,
        }
      : {
          deviceToken: this.deviceToken,
          update: dpUpdate,
          numExamples: y.length,
        };
    const response = await this.transport.postUpdate(body);

    return {
      sent: response.accepted !== false,
      numExamples: y.length,
      updateNorm: Math.sqrt(body.update.reduce((s, v) => s + v * v, 0)),
      localRecall: recall(trained, X, y),
      response,
    };
  }
}
