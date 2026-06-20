"use client";
import { useState } from "react";
import { api } from "../lib/api";
import type { PredictResponse } from "../lib/types";

interface ExplainResult {
  explanation: string;
  source?: string;
  groundedIn?: string;
}

// The single account both the scorer and the analyst rationale describe. The
// model scores this exact feature vector (see inspect()), so the indicators and
// the narration are genuinely about the same transaction.
const SAMPLE_TXN = { accountAgeDays: 2, fanout: 9, velocity: 1, campaignSignature: 1 };

export default function Inspector() {
  const [loading, setLoading] = useState(false);
  const [predict, setPredict] = useState<PredictResponse | null>(null);
  const [explain, setExplain] = useState<ExplainResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const inspect = async (): Promise<void> => {
    setLoading(true);
    setError(null);
    setExplain(null);
    let scored: PredictResponse | null = null;
    try {
      // Score the exact account we narrate below, so the displayed indicators
      // are the model's output for SAMPLE_TXN — not a different stub vector.
      scored = await api.predict(SAMPLE_TXN);
      setPredict(scored);
    } catch {
      setPredict(null);
      setError("Scoring service unreachable — start the API to inspect a live account.");
    }
    try {
      // Ground the LLM summary in the real model output: pass the model's
      // label / confidence / indicators so the summary can only restate them.
      const res = await fetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transaction: SAMPLE_TXN,
          label: scored?.label,
          confidence: scored?.confidence,
          indicators: scored?.indicators,
        }),
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
          {/* PRIMARY signal — the real model's /predict output. */}
          <div className="flex items-center justify-between gap-3">
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
            <span
              className="shrink-0 inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em]"
              style={{ background: "var(--bg-surface-2)", color: "var(--fed)" }}
            >
              <span aria-hidden className="h-1.5 w-1.5 rounded-full" style={{ background: "var(--fed)" }} />
              model output
            </span>
          </div>

          {predict.indicators?.length > 0 && (
            <div>
              <p className="eyebrow mb-1.5 text-text-muted">Indicators the model fired on</p>
              <ul className="space-y-1.5">
                {predict.indicators.map((ind) => (
                  <li key={ind} className="flex items-start gap-2 text-[13px] text-text-secondary">
                    <span aria-hidden className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-accent-gold" />
                    {ind}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {explain && (
        <figure
          className="mt-5 rounded-[14px] border-l-2 bg-bg-deep/60 p-4"
          style={{ borderColor: "var(--accent-gold)" }}
        >
          <figcaption className="eyebrow mb-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-text-muted">
            <span>Plain-English summary</span>
            <span aria-hidden>·</span>
            <span
              className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.12em]"
              style={{ background: "var(--bg-surface-2)", color: "var(--accent-gold)" }}
            >
              {explain.source === "minimax" ? "MiniMax-Text-01" : "offline fallback"}
            </span>
          </figcaption>
          <blockquote className="text-[13px] leading-relaxed text-text-secondary">
            {explain.explanation}
          </blockquote>
          {explain.groundedIn && (
            <p className="mt-2.5 text-[11px] leading-snug text-text-muted">
              Grounded in: <span className="text-text-secondary">{explain.groundedIn}</span>
            </p>
          )}
        </figure>
      )}

      <p className="mt-auto pt-5 text-[11px] leading-snug text-text-muted">
        The label, confidence and indicators are the model&rsquo;s own output. The summary only restates
        them in plain English — generated server-side, and the MiniMax key never reaches the browser.
      </p>
    </section>
  );
}
