import type { Bank, Regime, State } from "./types";

/** Mean detection across banks for a regime; 0 when there is no data yet. */
export function meanDetection(banks: Bank[] | undefined, regime: Regime): number {
  if (!banks || banks.length === 0) return 0;
  const sum = banks.reduce((acc, b) => acc + (b.detection?.[regime] ?? 0), 0);
  return sum / banks.length;
}

export function safeBanks(state: State | null): Bank[] {
  return state?.banks ?? [];
}
