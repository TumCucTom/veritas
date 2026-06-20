/**
 * On-device behavioural / payment events and the local training buffer.
 *
 * RAW EVENTS NEVER LEAVE THE DEVICE. They live only in this buffer and are
 * consumed locally to train the model. Only a DP-protected weight delta is
 * ever transmitted (see federation.ts).
 *
 * Feature vector order matches core/veritas_core/data.py FEATURE_DIM (10):
 *   [amount, oldOrig, newOrig, oldDest, newDest, accountAge,
 *    velocity, fanout, isTransfer, campaignSig]
 */
import { FEATURE_DIM } from "./model.js";
import type { Vector, Matrix } from "./model.js";
import { Rng } from "./rng.js";

/** A payment the host app asks the SDK to observe / score. */
export interface PaymentObservation {
  payeeId: string;
  /** Transfer amount in account currency units (e.g. GBP). */
  amount: number;
  /** First time this customer has paid this payee? */
  isNewPayee: boolean;
  /** Session signals the device can see (all optional). */
  remoteAccessAppActive?: boolean;
  inboundCallActive?: boolean;
  /** Behavioural-biometric deviation from this user's own baseline, 0..1. */
  sessionAnomaly?: number;
  /** Payments by this user in the last hour (velocity). */
  velocity1h?: number;
  /** Distinct payees in the last 24h (fan-out). */
  fanout24h?: number;
  /** Account age in days (older = more trusted). */
  accountAgeDays?: number;
}

/** A labelled event sitting in the on-device buffer. */
export interface DeviceEvent {
  features: Vector; // length FEATURE_DIM
  label: 0 | 1; // 1 = confirmed/suspected fraud (victim side)
  observedAt: number; // epoch ms — never transmitted
}

/**
 * Map a host-app payment into the model's feature contract.
 *
 * Features are standardised-ish (centred near 0) so they sit on the same
 * scale as the synthetically generated training data and the bank's model.
 */
export function paymentToFeatures(p: PaymentObservation): Vector {
  const amount = p.amount;
  // log-scale amount, centred so a "typical" ~£200 payment is near 0.
  const amountZ = Math.log10(Math.max(1, amount)) - 2.3;
  const accountAgeDays = p.accountAgeDays ?? 365;
  const accountAgeZ = (Math.log10(Math.max(1, accountAgeDays)) - 2.5) * 1.5;
  const velocity = p.velocity1h ?? 0;
  const velocityZ = velocity - 1;
  const fanout = p.fanout24h ?? 0;
  const fanoutZ = fanout - 1;
  const anomaly = clamp01(p.sessionAnomaly ?? 0);

  // oldOrig/newOrig/oldDest/newDest balance deltas aren't device-visible;
  // we proxy them from amount + new-payee + coercion session signals so the
  // on-device feature vector still occupies the same 10-dim space.
  const coercion =
    (p.remoteAccessAppActive ? 1 : 0) + (p.inboundCallActive ? 1 : 0);
  const oldOrig = amountZ * 0.5;
  const newOrig = -amountZ * 0.5;
  const oldDest = p.isNewPayee ? -0.5 : 0.2;
  const newDest = p.isNewPayee ? 0.5 + anomaly : 0.0;

  return [
    amountZ, // 0 amount
    oldOrig, // 1 oldOrig
    newOrig, // 2 newOrig
    oldDest, // 3 oldDest
    newDest, // 4 newDest
    accountAgeZ, // 5 accountAge
    velocityZ + coercion * 0.5, // 6 velocity
    fanoutZ, // 7 fanout
    p.isNewPayee ? 1.0 : 0.0, // 8 isTransfer (new-payee transfer proxy)
    anomaly * 1.5, // 9 campaignSig (anomaly proxy)
  ];
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/**
 * Generate a synthetic on-device event buffer. Mirrors data.py make_bank_data:
 * Gaussian features, fraud positives shifted on amount[0], velocity[6],
 * fanout[7], plus a "scam-in-progress" subset that shifts the anomaly axis [9].
 */
export function generateSyntheticEvents(
  n: number,
  fraudRate: number,
  seed: number,
): DeviceEvent[] {
  const rng = new Rng(seed);
  const out: DeviceEvent[] = [];
  for (let i = 0; i < n; i++) {
    const features: Vector = [];
    for (let j = 0; j < FEATURE_DIM; j++) features.push(rng.normal(0, 1));
    const isFraud = rng.next() < fraudRate;
    if (isFraud) {
      features[0]! += 1.5; // amount up
      features[6]! += 1.5; // velocity up
      features[7]! += 1.5; // fanout up
      // half of fraud is a coercion / scam-in-progress pattern: anomaly axis up
      if (rng.next() < 0.5) features[9]! += 1.5;
    }
    out.push({
      features,
      label: isFraud ? 1 : 0,
      observedAt: Date.now() - i,
    });
  }
  return out;
}

/** Split a buffer into the X matrix and y vector for training. */
export function toTrainingSet(buf: DeviceEvent[]): { X: Matrix; y: Vector } {
  return {
    X: buf.map((e) => e.features),
    y: buf.map((e) => e.label),
  };
}

/** A bounded ring buffer for on-device events (oldest evicted). */
export class EventBuffer {
  private readonly cap: number;
  private events: DeviceEvent[] = [];

  constructor(capacity = 2000) {
    this.cap = capacity;
  }

  add(e: DeviceEvent): void {
    this.events.push(e);
    if (this.events.length > this.cap) {
      this.events.splice(0, this.events.length - this.cap);
    }
  }

  addMany(es: DeviceEvent[]): void {
    for (const e of es) this.add(e);
  }

  all(): DeviceEvent[] {
    return this.events.slice();
  }

  size(): number {
    return this.events.length;
  }
}
