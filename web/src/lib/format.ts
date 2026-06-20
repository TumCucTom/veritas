const GBP = new Intl.NumberFormat("en-GB", {
  style: "currency",
  currency: "GBP",
  maximumFractionDigits: 0,
});

const NUM = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });

const NOT_DETECTED_HOURS = 101;

export function formatGbp(value: number): string {
  return GBP.format(Math.round(value || 0));
}

/** Compact GBP for hero scale: £1.2m / £840k. */
export function formatGbpCompact(value: number): string {
  const v = Math.round(value || 0);
  if (v >= 1_000_000) return `£${(v / 1_000_000).toFixed(v >= 10_000_000 ? 0 : 1)}m`;
  if (v >= 1_000) return `£${(v / 1_000).toFixed(v >= 100_000 ? 0 : 0)}k`;
  return GBP.format(v);
}

export function formatNumber(value: number): string {
  return NUM.format(Math.round(value || 0));
}

/**
 * Detection latency, rendered with the SAME unit-aware formatter for both
 * regimes (federated and siloed): minutes under an hour, hours up to two days,
 * then concrete days — so siloed reads as e.g. "4.2d" instead of a vague
 * "days+". Only a genuinely-absent value falls back to an em dash.
 */
export function formatTimeToDetect(hours: number): string {
  if (!Number.isFinite(hours)) return "—";
  if (hours < 1) return `${Math.round(hours * 60)}m`;
  if (hours < 48) return `${hours.toFixed(hours < 10 ? 1 : 0)}h`;
  return `${(hours / 24).toFixed(1)}d`;
}

export function isDetected(hours: number): boolean {
  return Number.isFinite(hours) && hours < NOT_DETECTED_HOURS;
}
