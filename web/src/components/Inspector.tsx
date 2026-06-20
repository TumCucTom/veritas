"use client";
import { useState } from "react";
import { api } from "../lib/api";

interface PredictResult {
  label: string;
  confidence: number;
  indicators: string[];
}

interface ExplainResult {
  explanation: string;
  source?: string;
}

const SAMPLE_TXN = { accountAgeDays: 2, fanout: 9, velocity: 1, campaignSignature: 1 };

export default function Inspector() {
  const [loading, setLoading] = useState(false);
  const [predict, setPredict] = useState<PredictResult | null>(null);
  const [explain, setExplain] = useState<ExplainResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const inspect = async (): Promise<void> => {
    setLoading(true);
    setError(null);
    setExplain(null);
    try {
      const p = (await api.predict({ campaignSignature: 1 })) as PredictResult;
      setPredict(p);
    } catch {
      setPredict(null);
      setError("Scoring service unreachable — start the API to inspect a live account.");
    }
    try {
      const res = await fetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transaction: SAMPLE_TXN }),
      });
      const data = (await res.json()) as ExplainResult;
      setExplain(data);
    } catch {
      setExplain(null);
    } finally {
      setLoading(false);
    }
  };

  const isFraud = predict?.label === "fraud";

  return (
    <section
      aria-labelledby="inspector-heading"
      className="flex h-full flex-col rounded-[20px] border bg-bg-surface/70 p-5 backdrop-blur-sm sm:p-6"
      style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-raise)" }}
    >
      <header className="mb-4 flex items-start justify-between gap-3">
        <div>
          <p className="eyebrow text-accent-gold">Analyst view</p>
          <h2
            id="inspector-heading"
            className="font-display text-xl tracking-tight text-text-primary"
          >
            Inspect a flagged account
          </h2>
        </div>
      </header>

      <button
        type="button"
        onClick={() => void inspect()}
        disabled={loading}
        className="inline-flex w-fit items-center gap-2 rounded-[11px] border px-4 py-2.5 text-[13px] font-semibold text-text-primary transition-all duration-200 ease-[cubic-bezier(0.16,1,0.3,1)] hover:-translate-y-px hover:border-accent-gold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-gold focus-visible:ring-offset-2 focus-visible:ring-offset-bg-deep disabled:cursor-not-allowed disabled:opacity-50"
        style={{ borderColor: "var(--border-strong)", background: "var(--bg-surface-2)" }}
      >
        {loading ? "Inspecting…" : "Inspect flagged account"}
      </button>

      {error && <p className="mt-3 text-[12px]" style={{ color: "var(--silo)" }}>{error}</p>}

      {predict && (
        <div className="mt-5 space-y-4">
          <div className="flex items-center gap-3">
            <span
              className="rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.12em]"
              style={{
                background: isFraud ? "rgba(240,74,82,0.14)" : "rgba(52,211,153,0.14)",
                color: isFraud ? "var(--silo)" : "var(--fed)",
              }}
            >
              {predict.label}
            </span>
            <span className="tabular text-[13px] text-text-secondary">
              {Math.round((predict.confidence ?? 0) * 100)}% confidence
            </span>
          </div>

          {predict.indicators?.length > 0 && (
            <ul className="space-y-1.5">
              {predict.indicators.map((ind) => (
                <li key={ind} className="flex items-start gap-2 text-[13px] text-text-secondary">
                  <span aria-hidden className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-accent-gold" />
                  {ind}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {explain && (
        <figure
          className="mt-5 rounded-[14px] border-l-2 bg-bg-deep/60 p-4"
          style={{ borderColor: "var(--accent-gold)" }}
        >
          <blockquote className="font-display text-[15px] leading-relaxed text-text-primary">
            “{explain.explanation}”
          </blockquote>
          <figcaption className="eyebrow mt-3">
            {explain.source === "minimax" ? "MiniMax-Text-01" : "Offline fallback"} · analyst rationale
          </figcaption>
        </figure>
      )}

      <p className="mt-auto pt-5 text-[11px] leading-snug text-text-muted">
        The explanation is generated server-side. The MiniMax key never reaches the browser.
      </p>
    </section>
  );
}
