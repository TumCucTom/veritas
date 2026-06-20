/**
 * Scam-in-progress signal — a small local rule layer that combines the
 * on-device model score with behavioural flags the device can directly see,
 * producing a human-readable reason for the verdict.
 *
 * This is the §3.1 "scam-in-progress" detector: new payee + high amount +
 * anomalous session + coercion signals (remote-access / overlapping call).
 * It never sends anything; it only explains the local verdict.
 */
import type { PaymentObservation } from "./events.js";

export interface RiskVerdict {
  /** Final risk in [0, 1]. */
  risk: number;
  /** Plain-English rationale (§6 explainability, on-device only). */
  reason: string;
  /** Suggested action for the host app. */
  action: "allow" | "review" | "step_up" | "hold";
  /** Structured flags that fired (useful for the host app / analytics). */
  indicators: string[];
}

const HIGH_AMOUNT = 1000;
const VERY_HIGH_AMOUNT = 5000;

/**
 * Combine the model probability with behavioural flags. The model score is the
 * base; coercion / new-payee / high-amount signals can escalate it, mirroring
 * how a victim-side detector treats socially-engineered "safe account" flows.
 */
export function assessScam(
  modelScore: number,
  p: PaymentObservation,
): RiskVerdict {
  const indicators: string[] = [];
  let risk = modelScore;

  const anomaly = clamp01(p.sessionAnomaly ?? 0);

  if (p.isNewPayee) {
    indicators.push("new_payee");
    risk += 0.1;
  }
  if (p.amount >= VERY_HIGH_AMOUNT) {
    indicators.push("very_high_amount");
    risk += 0.2;
  } else if (p.amount >= HIGH_AMOUNT) {
    indicators.push("high_amount");
    risk += 0.1;
  }
  if (anomaly >= 0.5) {
    indicators.push("anomalous_session");
    risk += 0.15 * anomaly;
  }
  if (p.remoteAccessAppActive) {
    indicators.push("remote_access_active");
    risk += 0.25;
  }
  if (p.inboundCallActive) {
    indicators.push("call_overlap");
    risk += 0.15;
  }

  // The classic coercion combo: a first-time high-value transfer while a
  // remote-access tool is live and an inbound call overlaps. Hard escalation.
  const coercionCombo =
    p.isNewPayee &&
    p.amount >= HIGH_AMOUNT &&
    (p.remoteAccessAppActive || p.inboundCallActive);
  if (coercionCombo) {
    indicators.push("coercion_pattern");
    risk = Math.max(risk, 0.85);
  }

  risk = clamp01(risk);

  let action: RiskVerdict["action"];
  if (risk >= 0.8) action = "hold";
  else if (risk >= 0.6) action = "step_up";
  else if (risk >= 0.35) action = "review";
  else action = "allow";

  return { risk, reason: buildReason(risk, indicators, p), action, indicators };
}

function buildReason(
  risk: number,
  indicators: string[],
  p: PaymentObservation,
): string {
  if (indicators.includes("coercion_pattern")) {
    return `Possible scam in progress: first-time payee for £${fmt(
      p.amount,
    )} while remote-access / a live call is active — classic authorised-push-payment coercion. Recommend pausing and verifying out-of-band.`;
  }
  if (risk >= 0.6) {
    const parts: string[] = [];
    if (indicators.includes("new_payee")) parts.push("a new payee");
    if (
      indicators.includes("high_amount") ||
      indicators.includes("very_high_amount")
    )
      parts.push(`an unusually high amount (£${fmt(p.amount)})`);
    if (indicators.includes("anomalous_session"))
      parts.push("an out-of-pattern session");
    const why = parts.length ? parts.join(", ") : "an out-of-pattern model signal";
    return `Elevated risk: ${why}. Recommend step-up verification before sending.`;
  }
  if (risk >= 0.35) {
    return "Some risk signals present; light review suggested.";
  }
  return "Looks consistent with this customer's normal behaviour.";
}

function fmt(n: number): string {
  return n.toLocaleString("en-GB", { maximumFractionDigits: 2 });
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
