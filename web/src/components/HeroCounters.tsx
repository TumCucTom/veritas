"use client";
import { useVeritas } from "../lib/store";
import { useCountUp } from "../lib/useCountUp";
import {
  formatGbpCompact,
  formatNumber,
  formatTimeToDetect,
} from "../lib/format";

export default function HeroCounters() {
  const { state } = useVeritas();

  const fed = state?.counters?.federated;
  const silo = state?.counters?.siloed;

  const preventedTarget = fed?.fraudPreventedGbp ?? 0;
  const victimsSparedTarget = Math.max(0, (silo?.victims ?? 0) - (fed?.victims ?? 0));

  const prevented = useCountUp(preventedTarget);
  const victimsSpared = useCountUp(victimsSparedTarget);

  const fedTtd = fed?.timeToDetectHours ?? Infinity;
  const siloTtd = silo?.timeToDetectHours ?? Infinity;

  return (
    <section
      aria-label="Federated impact"
      className="rise grid grid-cols-1 gap-px overflow-hidden rounded-[20px] border md:grid-cols-3"
      style={{
        borderColor: "var(--border-default)",
        background: "var(--border-default)",
        boxShadow: "var(--shadow-panel)",
      }}
    >
      {/* Hero stat — fraud prevented. Strongest scale + gold accent. */}
      <div className="bg-bg-surface p-6 sm:p-8">
        <p className="eyebrow text-accent-gold">Fraud prevented · Veritas</p>
        <p
          className="tabular mt-3 font-display text-[clamp(2.75rem,2rem+4vw,5rem)] font-semibold leading-[0.95] tracking-tight text-text-primary"
          style={{ textShadow: "0 0 40px rgba(214,168,91,0.18)" }}
        >
          {formatGbpCompact(prevented)}
        </p>
        <p className="mt-2 text-[13px] text-text-secondary">
          versus the siloed status quo, this campaign
        </p>
      </div>

      {/* Victims spared */}
      <div className="bg-bg-surface p-6 sm:p-8">
        <p className="eyebrow" style={{ color: "var(--fed)" }}>
          Victims spared
        </p>
        <p className="tabular mt-3 font-display text-[clamp(2.25rem,1.5rem+3vw,3.75rem)] font-semibold leading-[0.95] tracking-tight text-text-primary">
          {formatNumber(victimsSpared)}
        </p>
        <p className="mt-2 text-[13px] text-text-secondary">
          customers protected by shared intelligence
        </p>
      </div>

      {/* Time-to-detect race */}
      <div className="bg-bg-surface p-6 sm:p-8">
        <p className="eyebrow">Time to detect</p>
        <div className="mt-3 flex items-end gap-4">
          <div>
            <p
              className="tabular font-display text-[clamp(2rem,1.4rem+2.4vw,3.25rem)] font-semibold leading-[0.95] tracking-tight"
              style={{ color: "var(--fed)" }}
            >
              {formatTimeToDetect(fedTtd)}
            </p>
            <p className="eyebrow mt-1" style={{ color: "var(--fed)" }}>
              Federated
            </p>
          </div>
          <span className="mb-3 text-text-muted">vs</span>
          <div>
            <p
              className="tabular font-display text-[clamp(1.5rem,1.1rem+1.6vw,2.25rem)] font-semibold leading-[0.95] tracking-tight"
              style={{ color: "var(--silo)" }}
            >
              {formatTimeToDetect(siloTtd)}
            </p>
            <p className="eyebrow mt-1" style={{ color: "var(--silo)" }}>
              Siloed
            </p>
          </div>
        </div>
        <p className="mt-2 text-[13px] text-text-secondary">
          how fast the network reacts to a live campaign
        </p>
      </div>
    </section>
  );
}
