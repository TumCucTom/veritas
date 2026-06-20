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

/** Detection latency: undetected windows surface as days, not hours. */
export function formatTimeToDetect(hours: number): string {
  if (!Number.isFinite(hours) || hours >= NOT_DETECTED_HOURS) return "days+";
  if (hours < 1) return `${Math.round(hours * 60)}m`;
  return `${hours.toFixed(hours < 10 ? 1 : 0)}h`;
}

export function isDetected(hours: number): boolean {
  return Number.isFinite(hours) && hours < NOT_DETECTED_HOURS;
}
