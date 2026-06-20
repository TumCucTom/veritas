"use client";
import PopulationCanvas, { POPULATION_DOTS } from "./PopulationCanvas";
import { useVeritas } from "../lib/store";
import { meanDetection } from "../lib/derive";
import { formatGbp, formatNumber, formatTimeToDetect } from "../lib/format";
import type { Regime } from "../lib/types";

const COPY: Record<
  Regime,
  { eyebrow: string; title: string; sub: string; tone: "fed" | "silo" }
> = {
  siloed: {
    eyebrow: "Today",
    title: "Siloed banks",
    sub: "Each institution fights fraud alone. The campaign spreads bank to bank.",
    tone: "silo",
  },
  federated: {
    eyebrow: "Veritas",
    title: "Federated network",
    sub: "Banks share fraud intelligence — not customer data. Protection compounds.",
    tone: "fed",
  },
};

interface Props {
  regime: Regime;
}

export default function RacePanel({ regime }: Props) {
  const { state } = useVeritas();
  const copy = COPY[regime];
  const isFed = regime === "federated";

  const banks = state?.banks ?? [];
  const avg = meanDetection(banks, regime);
  const counters = state?.counters?.[regime];

  const lost = counters?.lostGbp ?? 0;
  const victims = counters?.victims ?? 0;
  const ttd = counters?.timeToDetectHours ?? Infinity;

  const accent = isFed ? "var(--fed)" : "var(--silo)";
  const pct = Math.round(avg * 100);

  // Honest dot ratio: the canvas paints POPULATION_DOTS dots for the whole
  // customer base, so each dot stands for (total customers / dots). Derived
  // from live state rather than a hardcoded figure so the caption can't drift.
  const totalCustomers = banks.reduce((sum, b) => sum + (b.customers ?? 0), 0);
  const customersPerDot = totalCustomers > 0 ? Math.round(totalCustomers / POPULATION_DOTS) : 0;

  return (
    <section
      aria-labelledby={`race-${regime}`}
      className="relative flex flex-col rounded-[20px] border bg-bg-surface/80 p-5 backdrop-blur-sm sm:p-6"
      style={{
        borderColor: isFed ? "rgba(52,211,153,0.28)" : "rgba(240,74,82,0.24)",
        boxShadow: "var(--shadow-panel)",
      }}
    >
      {/* Accent rail */}
      <span
        aria-hidden
        className="absolute left-0 top-6 bottom-6 w-[3px] rounded-full"
        style={{ background: accent, boxShadow: `0 0 18px ${accent}` }}
      />

      <header className="mb-4 flex items-start justify-between gap-4 pl-3">
        <div>
          <p className="eyebrow" style={{ color: accent }}>
            {copy.eyebrow}
          </p>
          <h2
            id={`race-${regime}`}
            className="font-display text-2xl leading-tight tracking-tight text-text-primary sm:text-[28px]"
          >
            {copy.title}
          </h2>
          <p className="mt-1 max-w-sm text-[13px] leading-snug text-text-secondary">
            {copy.sub}
          </p>
        </div>
        <div className="shrink-0 text-right">
          <p className="tabular font-display text-3xl leading-none" style={{ color: accent }}>
            {pct}%
          </p>
          <p className="eyebrow mt-1">detected</p>
        </div>
      </header>

      <div
        className="relative overflow-hidden rounded-[14px] border"
        style={{ borderColor: "var(--border-default)", background: "var(--bg-deep)" }}
      >
        <PopulationCanvas detection={avg} tone={copy.tone} />
        {!state && (
          <div className="absolute inset-0 grid place-items-center text-[12px] text-text-muted">
            awaiting network…
          </div>
        )}
      </div>
      <p className="mt-2 pl-1 text-[11px] text-text-muted">
        {customersPerDot > 0 ? `1 dot ≈ ${formatNumber(customersPerDot)} customers · ` : ""}red =
        victimised · green = protected
      </p>

      <div className="hairline my-4" />

      <dl className="grid grid-cols-3 gap-3 pl-1">
        <Stat label="Lost to fraud" value={formatGbp(lost)} accent={accent} emphasise={!isFed} />
        <Stat label="Victims" value={formatNumber(victims)} accent={accent} emphasise={!isFed} />
        <Stat
          label="Time to detect"
          value={formatTimeToDetect(ttd)}
          accent={accent}
          emphasise={isFed}
        />
      </dl>
    </section>
  );
}

function Stat({
  label,
  value,
  accent,
  emphasise,
}: {
  label: string;
  value: string;
  accent: string;
  emphasise: boolean;
}) {
  return (
    <div>
      <dt className="eyebrow mb-1.5">{label}</dt>
      <dd
        className="tabular font-display text-xl leading-none tracking-tight sm:text-2xl"
        style={{ color: emphasise ? accent : "var(--text-primary)" }}
      >
        {value}
      </dd>
    </div>
  );
}
