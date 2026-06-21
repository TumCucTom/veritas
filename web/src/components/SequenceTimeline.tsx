"use client";

import { memo, useEffect, useMemo, useRef, useState } from "react";

// ── Data schema (web/public/viz/sequence.json) ─────────────────────────────
type SeqStep = {
  t: number;
  amount: number;
  hour: number;
  fanout?: number;
  score: number; // 0..1, running GRU fraud probability after this step
  note?: string;
};

type Sequence = {
  id: string;
  label: 0 | 1; // 1 = mule/fraud, 0 = legit
  steps: SeqStep[];
};

type SequenceData = {
  sequences: Sequence[];
  meta: { note: string; model: string };
};

// Tuning ────────────────────────────────────────────────────────────────────
const STEP_MS = 520; // ~0.5s per transaction step
const END_PAUSE_MS = 1400; // dwell on the last step before switching sequence
const THRESHOLD = 0.5; // GRU "decides mule" once running score crosses this

// Human-readable per-sequence caption + display name.
const SEQ_META: Record<string, { name: string; blurb: string }> = {
  "mule-layering": {
    name: "Suspected mule",
    blurb:
      "Quiet, dormant account — then a sudden burst of high-value, high-fan-out transfers. The GRU reads the pattern and the score climbs past threshold.",
  },
  "salary-and-bills": {
    name: "Salary & bills",
    blurb:
      "Ordinary inflows and outflows over time. No burst, no fan-out — the running score never lifts off the floor.",
  },
  "busy-but-honest": {
    name: "Busy but honest",
    blurb:
      "A few larger payments mid-month wobble the score, but there's no layering pattern, so the GRU settles it back down.",
  },
};

function clamp01(v: number) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}
function easeInOut(t: number) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}
function fmtAmount(v: number) {
  if (v >= 1000) return `£${(v / 1000).toFixed(1)}k`;
  return `£${Math.round(v)}`;
}
function fmtHour(h: number) {
  const hh = ((h % 24) + 24) % 24;
  return `${String(hh).padStart(2, "0")}:00`;
}

function SequenceTimelineInner() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  const [data, setData] = useState<SequenceData | null>(null);
  const [error, setError] = useState(false);

  // UI-facing state, written at a throttled cadence from the RAF loop.
  const [seqIdx, setSeqIdx] = useState(0);
  const [activeStep, setActiveStep] = useState(0);
  const [liveScore, setLiveScore] = useState(0);
  const [paused, setPaused] = useState(false);

  // Manual sequence override (fraud/legit toggle), read inside the loop via ref.
  const seqOverrideRef = useRef<number | null>(null);

  // ── Fetch own data ───────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    fetch("/viz/sequence.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((json: SequenceData) => {
        if (cancelled) return;
        if (!json?.sequences?.length) throw new Error("empty");
        setData(json);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const seqCount = data?.sequences.length ?? 0;

  // ── Single RAF render loop, held in refs so it isn't recreated on render ──
  useEffect(() => {
    if (!data) return;
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const sequences = data.sequences;

    let width = 0;
    let height = 0;
    let dpr = 1;

    const sizeCanvas = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = wrap.getBoundingClientRect();
      width = Math.max(320, Math.floor(rect.width));
      height = Math.max(220, Math.floor(rect.height));
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    sizeCanvas();

    let raf = 0;
    let visible = true;
    const pausedLocal = paused;

    // Playback state lives in the loop (no per-frame allocation).
    let curSeq = seqOverrideRef.current ?? seqIdx;
    let fromStep = reduceMotion ? sequences[curSeq].steps.length - 1 : 0;
    let stepStart = performance.now();

    let lastUiSeq = -1;
    let lastUiStep = -1;
    let lastUiScoreBucket = -1;

    const draw = (now: number) => {
      // ── Resolve active sequence (toggle override wins) ────────────────────
      const wantSeq = seqOverrideRef.current;
      if (wantSeq !== null && wantSeq !== curSeq) {
        curSeq = wantSeq;
        fromStep = reduceMotion ? sequences[curSeq].steps.length - 1 : 0;
        stepStart = now;
      }

      const seq = sequences[curSeq];
      const steps = seq.steps;
      const lastStep = steps.length - 1;

      // ── Advance the step cursor ───────────────────────────────────────────
      let toStep: number;
      let phase: number;
      if (pausedLocal || reduceMotion) {
        toStep = fromStep;
        phase = 1;
        stepStart = now;
      } else {
        const dwell = fromStep === lastStep ? END_PAUSE_MS : STEP_MS;
        const raw = (now - stepStart) / dwell;
        if (raw >= 1) {
          if (fromStep >= lastStep) {
            // loop / advance to the next sequence (only when not toggled-locked)
            if (seqOverrideRef.current === null) {
              curSeq = (curSeq + 1) % sequences.length;
            }
            fromStep = 0;
          } else {
            fromStep += 1;
          }
          stepStart = now;
          toStep = fromStep;
          phase = 0;
        } else {
          toStep = fromStep;
          phase = easeInOut(clamp01(raw));
        }
      }

      // Score revealed so far: solid up to fromStep, easing into the current.
      const revealed = clamp01(fromStep + phase); // fractional step revealed

      // ── Geometry ──────────────────────────────────────────────────────────
      const PADX = 30;
      const TOP = 26;
      const railY = Math.round(height * 0.30); // transaction dots rail
      const chartTop = railY + 34;
      const chartBot = height - 30;
      const chartH = Math.max(40, chartBot - chartTop);
      const n = steps.length;
      const innerW = width - PADX * 2;
      const stepX = (i: number) => PADX + (n <= 1 ? 0.5 : i / (n - 1)) * innerW;
      const scoreY = (s: number) => chartBot - clamp01(s) * chartH;

      const isFraud = seq.label === 1;
      // warm gold for the fraud thread, calm teal for legit
      const hue = isFraud ? 40 : 188;

      // ── Background ────────────────────────────────────────────────────────
      ctx.clearRect(0, 0, width, height);
      const bg = ctx.createLinearGradient(0, 0, width, height);
      bg.addColorStop(0, "rgba(8,12,20,0.96)");
      bg.addColorStop(1, "rgba(16,23,38,0.96)");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, width, height);

      // ── Threshold band + line ─────────────────────────────────────────────
      const thrY = scoreY(THRESHOLD);
      ctx.fillStyle = "rgba(246,200,120,0.05)";
      ctx.fillRect(PADX, chartTop, innerW, Math.max(0, thrY - chartTop));
      ctx.setLineDash([4, 5]);
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(246,200,120,0.40)";
      ctx.beginPath();
      ctx.moveTo(PADX, thrY);
      ctx.lineTo(width - PADX, thrY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillStyle = "rgba(246,200,120,0.62)";
      ctx.fillText("decision threshold", PADX + 2, thrY - 3);

      // ── Timeline rail ─────────────────────────────────────────────────────
      ctx.strokeStyle = "rgba(120,140,170,0.20)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(PADX, railY);
      ctx.lineTo(width - PADX, railY);
      ctx.stroke();
      ctx.fillStyle = "rgba(150,170,200,0.45)";
      ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText("TRANSACTION TIMELINE", PADX, TOP - 18 < 0 ? 4 : TOP - 18);

      // ── Score line (only the revealed portion) ────────────────────────────
      ctx.lineWidth = 2.4;
      ctx.strokeStyle = `hsl(${hue},${isFraud ? 92 : 60}%,${isFraud ? 60 : 56}%)`;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < n; i++) {
        if (i > revealed) break;
        const x = stepX(i);
        const y = scoreY(steps[i].score);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      // partial segment into the in-progress step for smooth growth
      const fi = Math.floor(revealed);
      if (fi < n - 1 && revealed > fi) {
        const f = revealed - fi;
        const x = lerp(stepX(fi), stepX(fi + 1), f);
        const y = lerp(scoreY(steps[fi].score), scoreY(steps[fi + 1].score), f);
        ctx.lineTo(x, y);
      }
      if (started) ctx.stroke();

      // soft fill under the revealed line
      if (started) {
        const headX =
          fi < n - 1 && revealed > fi
            ? lerp(stepX(fi), stepX(fi + 1), revealed - fi)
            : stepX(Math.min(fi, n - 1));
        ctx.lineTo(headX, chartBot);
        ctx.lineTo(stepX(0), chartBot);
        ctx.closePath();
        const fill = ctx.createLinearGradient(0, chartTop, 0, chartBot);
        fill.addColorStop(0, `hsla(${hue},${isFraud ? 90 : 55}%,58%,0.22)`);
        fill.addColorStop(1, `hsla(${hue},${isFraud ? 90 : 55}%,58%,0)`);
        ctx.fillStyle = fill;
        ctx.fill();
      }

      // ── Transaction dots + connectors on the rail ─────────────────────────
      let curScore = steps[0].score;
      for (let i = 0; i < n; i++) {
        const x = stepX(i);
        const seen = i <= revealed + 0.001;
        const st = steps[i];
        const r = 4 + (st.fanout ? Math.min(st.fanout, 12) * 0.5 : 0);
        // amount drives a faint bar above the rail
        const barH = clamp01(st.amount / 3500) * 22;
        ctx.fillStyle = seen
          ? `hsla(${hue},${isFraud ? 70 : 45}%,60%,0.35)`
          : "rgba(120,140,170,0.10)";
        ctx.fillRect(x - 2, railY - barH, 4, barH);

        // dot
        if (seen) {
          ctx.fillStyle = `hsl(${hue},${isFraud ? 85 : 50}%,58%)`;
        } else {
          ctx.fillStyle = "rgba(120,140,170,0.28)";
        }
        ctx.beginPath();
        ctx.arc(x, railY, seen ? r : 3, 0, Math.PI * 2);
        ctx.fill();

        if (seen) curScore = st.score;
      }

      // ── Moving playhead marker on the score line ──────────────────────────
      const headStep = Math.min(Math.round(revealed), n - 1);
      const hx = stepX(headStep);
      const hy = scoreY(steps[headStep].score);
      const crossed = steps[headStep].score >= THRESHOLD;
      ctx.beginPath();
      ctx.arc(hx, hy, 5, 0, Math.PI * 2);
      ctx.fillStyle = crossed ? "hsl(40,95%,62%)" : `hsl(${hue},70%,62%)`;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "rgba(246,233,205,0.85)";
      ctx.stroke();
      if (crossed) {
        const g = ctx.createRadialGradient(hx, hy, 0, hx, hy, 16);
        g.addColorStop(0, "hsla(40,95%,62%,0.5)");
        g.addColorStop(1, "hsla(40,95%,62%,0)");
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(hx, hy, 16, 0, Math.PI * 2);
        ctx.fill();
      }

      // ── Annotate the active transaction (amount / hour / fanout) ──────────
      const aStep = steps[headStep];
      ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
      ctx.textBaseline = "bottom";
      ctx.textAlign = hx > width - 90 ? "right" : "left";
      const tx = hx > width - 90 ? hx - 8 : hx + 8;
      ctx.fillStyle = "rgba(228,236,248,0.92)";
      ctx.fillText(fmtAmount(aStep.amount), tx, railY - 6);
      ctx.font = "500 10px ui-sans-serif, system-ui, sans-serif";
      ctx.fillStyle = "rgba(160,178,205,0.78)";
      const meta = `${fmtHour(aStep.hour)}${aStep.fanout ? ` · ${aStep.fanout} payees` : ""}`;
      ctx.fillText(meta, tx, railY + 16);

      // ── Throttled UI sync ────────────────────────────────────────────────
      if (curSeq !== lastUiSeq) {
        lastUiSeq = curSeq;
        setSeqIdx(curSeq);
      }
      if (headStep !== lastUiStep) {
        lastUiStep = headStep;
        setActiveStep(headStep);
      }
      const bucket = Math.round(curScore * 100);
      if (bucket !== lastUiScoreBucket) {
        lastUiScoreBucket = bucket;
        setLiveScore(curScore);
      }

      if (visible) raf = requestAnimationFrame(draw);
    };

    const start = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(draw);
    };
    const stop = () => cancelAnimationFrame(raf);

    const io = new IntersectionObserver(
      (entries) => {
        const e = entries[0];
        visible = e.isIntersecting;
        if (visible && !pausedLocal) start();
        else stop();
      },
      { threshold: 0.05 },
    );
    io.observe(wrap);

    let firstResize = true;
    const ro = new ResizeObserver(() => {
      if (firstResize) {
        firstResize = false;
        return;
      }
      sizeCanvas();
    });
    ro.observe(wrap);

    start();

    return () => {
      stop();
      io.disconnect();
      ro.disconnect();
    };
    // Re-create loop when data ready or pause toggles.
  }, [data, paused, seqIdx]);

  const activeSeq = data?.sequences[seqIdx];
  const activeMeta = activeSeq ? SEQ_META[activeSeq.id] : undefined;
  const isFraud = activeSeq?.label === 1;
  const crossed = liveScore >= THRESHOLD;
  const caption =
    activeSeq?.steps[activeStep]?.note ??
    activeMeta?.blurb ??
    "The GRU reads the transaction sequence over time.";

  // ── Fallback ────────────────────────────────────────────────────────────
  if (error) {
    return (
      <section
        aria-label="Sequence model timeline"
        className="rise overflow-hidden rounded-[20px] border bg-bg-surface/80 p-6 backdrop-blur-sm sm:p-8"
        style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-panel)" }}
      >
        <p className="eyebrow text-accent-gold">Sequence model · GRU over time</p>
        <h2 className="mt-2 font-display text-[clamp(1.5rem,1.1rem+1.6vw,2.4rem)] leading-tight tracking-tight text-text-primary">
          Fraud is a pattern, not a transaction.
        </h2>
        <p className="mt-4 text-[13px] leading-relaxed text-text-secondary">
          Timeline telemetry is unavailable right now. The recurrent GRU still reads each
          customer&apos;s transaction sequence in time order — this visualization will resume once
          the data feed returns.
        </p>
      </section>
    );
  }

  return (
    <section
      aria-labelledby="sequence-timeline-heading"
      className="rise overflow-hidden rounded-[20px] border bg-bg-surface/80 backdrop-blur-sm"
      style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-panel)" }}
    >
      <header
        className="grid gap-4 border-b p-5 sm:p-6 lg:grid-cols-[1fr_auto] lg:items-end"
        style={{ borderColor: "var(--border-default)" }}
      >
        <div>
          <p className="eyebrow text-accent-gold">Sequence model · GRU over time</p>
          <h2
            id="sequence-timeline-heading"
            className="mt-2 font-display text-[clamp(1.7rem,1.2rem+2vw,3rem)] leading-tight tracking-tight text-text-primary"
          >
            Fraud is a pattern, not a transaction.
          </h2>
          <p className="mt-3 max-w-3xl text-[13px] leading-relaxed text-text-secondary sm:text-[14px]">
            A from-scratch GRU reads a customer&apos;s transactions in time order. No single payment
            is damning — but the mule&apos;s quiet-then-burst layering signature makes the running
            fraud score climb past threshold, while honest accounts stay flat and low.
          </p>
        </div>
        <div
          className="grid grid-cols-2 gap-px overflow-hidden rounded-[14px] border text-center sm:min-w-[300px]"
          style={{ borderColor: "var(--border-default)", background: "var(--border-default)" }}
        >
          <Stat label="step" value={`${activeStep + 1}/${activeSeq?.steps.length ?? 8}`} tone="gold" />
          <Stat
            label="fraud score"
            value={liveScore.toFixed(2)}
            tone={crossed ? "gold" : "fed"}
          />
        </div>
      </header>

      <div className="p-4 sm:p-5">
        <div
          ref={wrapRef}
          className="relative h-[300px] w-full overflow-hidden rounded-[14px] border bg-bg-deep sm:h-[340px]"
          style={{ borderColor: "var(--border-default)" }}
        >
          <canvas
            ref={canvasRef}
            className="block h-full w-full"
            role="img"
            aria-label="A customer transaction timeline with a running GRU fraud score rising beneath it"
          />

          {/* Active-sequence badge */}
          <div
            className="pointer-events-none absolute right-3 top-3 rounded-[10px] border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.12em] backdrop-blur-sm"
            style={{
              borderColor: "var(--border-default)",
              background: "rgba(11,16,28,0.7)",
              color: isFraud ? "var(--accent-gold)" : "var(--fed)",
            }}
          >
            {activeMeta?.name ?? activeSeq?.id ?? "sequence"}
          </div>

          {/* Caption overlay */}
          <p
            className="pointer-events-none absolute bottom-3 left-3 right-3 text-[12px] leading-snug text-text-secondary sm:text-[13px]"
            style={{ textShadow: "0 1px 8px rgba(8,12,20,0.92)" }}
          >
            {caption}
          </p>
        </div>

        {/* Controls */}
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => setPaused((p) => !p)}
            className="rounded-full border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-text-secondary transition-colors hover:text-text-primary"
            style={{ borderColor: "var(--border-strong)", background: "var(--bg-surface-2)" }}
          >
            {paused ? "play" : "pause"}
          </button>

          <div className="flex items-center gap-1.5">
            {data?.sequences.map((s, i) => {
              const active = i === seqIdx;
              return (
                <button
                  key={s.id}
                  type="button"
                  aria-label={`Show ${SEQ_META[s.id]?.name ?? s.id}`}
                  aria-pressed={active}
                  onClick={() => {
                    seqOverrideRef.current = i;
                    setSeqIdx(i);
                    setPaused(false);
                  }}
                  className="rounded-full border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.1em] transition-colors"
                  style={{
                    borderColor: active ? "transparent" : "var(--border-strong)",
                    background: active
                      ? s.label === 1
                        ? "var(--accent-gold)"
                        : "var(--fed)"
                      : "var(--bg-surface-2)",
                    color: active ? "#0b101c" : "var(--text-secondary)",
                  }}
                >
                  {s.label === 1 ? "fraud" : "legit"}
                  {data.sequences.filter((x) => x.label === s.label).length > 1
                    ? ` ${data.sequences.filter((x, j) => x.label === s.label && j <= i).length}`
                    : ""}
                </button>
              );
            })}
          </div>

          {seqOverrideRef.current !== null && (
            <button
              type="button"
              onClick={() => {
                seqOverrideRef.current = null;
              }}
              className="rounded-full border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-text-muted transition-colors hover:text-text-primary"
              style={{ borderColor: "var(--border-strong)", background: "var(--bg-surface-2)" }}
            >
              auto-cycle
            </button>
          )}

          <p className="tabular ml-auto text-[12px] text-text-muted">
            {data?.meta.model ?? "GRU"} · running score after each transaction
          </p>
        </div>
      </div>
    </section>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "fed" | "gold";
}) {
  const color = tone === "fed" ? "var(--fed)" : "var(--accent-gold)";
  return (
    <div className="bg-bg-surface-2 px-4 py-3">
      <p className="eyebrow" style={{ color }}>
        {label}
      </p>
      <p className="tabular mt-1 font-display text-2xl leading-none text-text-primary">{value}</p>
    </div>
  );
}

const SequenceTimeline = memo(SequenceTimelineInner);
export default SequenceTimeline;
