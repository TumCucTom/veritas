"use client";
import { meanDetection } from "../lib/derive";
import { useVeritas } from "../lib/store";

const STEPS = [
  "local train",
  "DP clip/noise",
  "Multi-Krum aggregate",
  "provenance recorded",
] as const;

export default function TechnicalProofStrip() {
  const { state, lastAttack } = useVeritas();
  const banks = state?.banks ?? [];
  const round = state?.round ?? 0;
  const fed = Math.round(meanDetection(banks, "federated") * 100);
  const silo = Math.round(meanDetection(banks, "siloed") * 100);
  const records = state?.customerRecordsTransmitted ?? 0;
  const attackRejected = Boolean(lastAttack?.rejected || state?.attackActive);

  return (
    <section
      aria-label="Technical proof"
      className="grid gap-px overflow-hidden rounded-[18px] border bg-border-default md:grid-cols-[1fr_auto]"
      style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-raise)" }}
    >
      <div className="bg-bg-surface/85 p-4 sm:p-5">
        <div className="flex flex-wrap items-center gap-2">
          {STEPS.map((step, index) => (
            <ProofStep
              key={step}
              label={step}
              active={round > 0 || index === 0}
              flagged={step === "Multi-Krum aggregate" && attackRejected}
            />
          ))}
        </div>
      </div>
      <div className="grid grid-cols-3 gap-px bg-border-default text-center md:min-w-[420px]">
        <ProofMetric label="round" value={String(round).padStart(2, "0")} />
        <ProofMetric label="records moved" value={String(records)} accent="var(--fed)" />
        <ProofMetric label="detect gap" value={`${fed}% / ${silo}%`} accent="var(--accent-gold)" />
      </div>
    </section>
  );
}

function ProofStep({
  label,
  active,
  flagged,
}: {
  label: string;
  active: boolean;
  flagged: boolean;
}) {
  const color = flagged ? "var(--silo)" : active ? "var(--fed)" : "var(--text-muted)";
  return (
    <span className="inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.12em]"
      style={{
        borderColor: flagged ? "rgba(240,74,82,0.45)" : "var(--border-strong)",
        background: flagged ? "rgba(240,74,82,0.08)" : "var(--bg-surface-2)",
        color,
      }}
    >
      <span
        aria-hidden
        className="h-1.5 w-1.5 rounded-full"
        style={{ background: color, boxShadow: active ? `0 0 10px ${color}` : undefined }}
      />
      {label}
    </span>
  );
}

function ProofMetric({
  label,
  value,
  accent = "var(--text-primary)",
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div className="bg-bg-surface-2 px-4 py-3">
      <p className="eyebrow">{label}</p>
      <p className="tabular mt-1 font-display text-xl leading-none" style={{ color: accent }}>
        {value}
      </p>
    </div>
  );
}
