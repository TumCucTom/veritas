"use client";

import { memo, useEffect, useMemo, useRef, useState } from "react";

// ── Data schema (web/public/viz/ensemble.json) ─────────────────────────────
type ModelInfo = {
  key: string;
  name: string;
  metric: number; // standalone metric (recall) on held-out sample, 0..1
  metricLabel: string;
  weight: number; // meta-learner's learned coefficient (signed)
};

type Sample = {
  id: number;
  label: 0 | 1;
  baseScores: Record<string, number>; // key -> 0..1
  finalScore: number; // meta-combined, 0..1
};

type EnsembleData = {
  models: ModelInfo[];
  ensemble: { metric: number; metricLabel: string };
  samples: Sample[];
  meta: { note: string; metricName: "AUC" | "recall" };
};

// Tuning ──────────────────────────────────────────────────────────────────
const SAMPLE_MS = 1500; // dwell per sample
const FRAUD_HUE = 8; // warm rose — fraud / lit
const LEGIT_HUE = 168; // cool teal — legit / calm
const META_HUE = 42; // gold — meta + ensemble

function clamp01(v: number) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}
function easeInOut(t: number) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

function EnsembleStackInner() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  const [data, setData] = useState<EnsembleData | null>(null);
  const [error, setError] = useState(false);

  // UI-facing state, written at a throttled cadence from the RAF loop.
  const [activeSample, setActiveSample] = useState(0);
  const [paused, setPaused] = useState(false);

  // Manual scrub override: when set, the loop holds on this sample index.
  const scrubRef = useRef<number | null>(null);
  const [scrubbing, setScrubbing] = useState(false);

  // ── Fetch own data ────────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    fetch("/viz/ensemble.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((json: EnsembleData) => {
        if (cancelled) return;
        if (!json?.models?.length || !json?.samples?.length) throw new Error("empty");
        setData(json);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const sampleCount = data?.samples.length ?? 0;
  const bestSingle = useMemo(
    () => (data ? Math.max(...data.models.map((m) => m.metric)) : 0),
    [data],
  );

  // ── Precompute a static layout ONCE ───────────────────────────────────────
  // Base models stack as a column on the left; a META node sits centre-right;
  // the ENSEMBLE verdict node sits far right. Edges base→meta carry weight.
  // All positions are normalised [0,1], mapped to pixels at draw time.
  const precomp = useMemo(() => {
    if (!data) return null;
    const { models } = data;
    const nb = models.length;

    const nodeY = new Float32Array(nb);
    const top = 0.1;
    const bottom = 0.9;
    for (let i = 0; i < nb; i++) {
      nodeY[i] = nb === 1 ? 0.5 : lerp(top, bottom, i / (nb - 1));
    }
    const baseX = 0.16;
    const metaX = 0.58;
    const metaY = 0.5;
    const ensX = 0.86;

    // Edge weight magnitude → thickness/opacity, normalised by the max |weight|.
    const absW = models.map((m) => Math.abs(m.weight));
    const maxW = Math.max(1e-6, ...absW);
    const wNorm = new Float32Array(nb);
    const wSign = new Float32Array(nb);
    for (let i = 0; i < nb; i++) {
      wNorm[i] = absW[i] / maxW;
      wSign[i] = models[i].weight >= 0 ? 1 : -1;
    }

    // Reusable per-frame buffers (allocated once, never re-allocated in RAF).
    const litBase = new Float32Array(nb); // base score for the active sample
    const flow = new Float32Array(nb); // 0..1 contribution-flow progress

    return { nb, nodeY, baseX, metaX, metaY, ensX, wNorm, wSign, litBase, flow };
  }, [data]);

  // ── Single RAF render loop, held in refs so it isn't recreated on render ──
  useEffect(() => {
    if (!data || !precomp) return;
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const { nb, nodeY, baseX, metaX, metaY, ensX, wNorm, wSign, litBase, flow } = precomp;
    const { models, samples } = data;
    const lastSample = samples.length - 1;
    const keys = models.map((m) => m.key);

    let width = 0;
    let height = 0;
    let dpr = 1;

    const sizeCanvas = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = wrap.getBoundingClientRect();
      width = Math.max(320, Math.floor(rect.width));
      height = Math.max(280, Math.floor(rect.height));
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    sizeCanvas();

    let raf = 0;
    let visible = true;
    const pausedLocal = paused;
    let stepStart = performance.now();
    let fromSample = reduceMotion ? 0 : 0;

    let lastUiSample = -1;

    const draw = (now: number) => {
      const PAD_X = 26;
      const PAD_Y = 22;
      const iw = width - PAD_X * 2;
      const ih = height - PAD_Y * 2;
      const px = (v: number) => PAD_X + v * iw;
      const py = (v: number) => PAD_Y + v * ih;

      // ── Resolve which sample + animation phase ───────────────────────────
      const scrub = scrubRef.current;
      let phase: number; // 0..1 within the dwell
      if (scrub !== null) {
        fromSample = scrub;
        phase = 1;
        stepStart = now;
      } else if (pausedLocal || reduceMotion) {
        phase = 1;
        stepStart = now;
      } else {
        const raw = (now - stepStart) / SAMPLE_MS;
        if (raw >= 1) {
          fromSample = fromSample >= lastSample ? 0 : fromSample + 1;
          stepStart = now;
          phase = 0;
        } else {
          phase = clamp01(raw);
        }
      }

      const sample = samples[fromSample];
      const fraud = sample.label === 1;
      const verdictHue = fraud ? FRAUD_HUE : LEGIT_HUE;

      // Animation envelopes within a sample:
      //   0.00–0.45  base models light up with their score
      //   0.30–0.80  contributions flow along the edges into the meta node
      //   0.65–1.00  meta resolves the final verdict
      const eLight = easeInOut(clamp01(phase / 0.45));
      const eFlow = easeInOut(clamp01((phase - 0.3) / 0.5));
      const eVerdict = easeInOut(clamp01((phase - 0.65) / 0.35));
      for (let i = 0; i < nb; i++) {
        litBase[i] = (sample.baseScores[keys[i]] ?? 0) * eLight;
        flow[i] = eFlow;
      }
      const finalLit = sample.finalScore * eVerdict;

      // ── Background ───────────────────────────────────────────────────────
      ctx.clearRect(0, 0, width, height);
      const bg = ctx.createLinearGradient(0, 0, width, height);
      bg.addColorStop(0, "rgba(8,12,20,0.96)");
      bg.addColorStop(0.6, "rgba(11,16,28,0.96)");
      bg.addColorStop(1, "rgba(16,23,38,0.96)");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, width, height);

      const mx = px(metaX);
      const my = py(metaY);
      const ex = px(ensX);
      const ey = py(metaY);

      // ── Edges base→meta (thickness/opacity ∝ |meta weight|) ──────────────
      for (let i = 0; i < nb; i++) {
        const x1 = px(baseX) + 18;
        const y1 = py(nodeY[i]);
        const w = wNorm[i];
        // negative-weight edges read cool/grey (the stack DISTRUSTS them)
        const edgeHue = wSign[i] >= 0 ? META_HUE : 210;
        const cx = lerp(x1, mx, 0.5);
        ctx.lineWidth = lerp(0.6, 4.2, w);
        ctx.strokeStyle = `hsla(${edgeHue},${wSign[i] >= 0 ? 70 : 12}%,55%,${(
          0.12 +
          0.5 * w
        ).toFixed(3)})`;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.quadraticCurveTo(cx, lerp(y1, my, 0.5), mx, my);
        ctx.stroke();

        // ── flowing contribution pulse along the edge ─────────────────────
        // brightness keyed to base score × weight, position by eFlow.
        const contrib = litBase[i] * (0.35 + 0.65 * w);
        if (contrib > 0.05 && flow[i] > 0.01) {
          const t = flow[i];
          const bx = (1 - t) * (1 - t) * x1 + 2 * (1 - t) * t * cx + t * t * mx;
          const by =
            (1 - t) * (1 - t) * y1 +
            2 * (1 - t) * t * lerp(y1, my, 0.5) +
            t * t * my;
          const pr = 2 + contrib * 5;
          const g = ctx.createRadialGradient(bx, by, 0, bx, by, pr * 2.4);
          g.addColorStop(0, `hsla(${META_HUE},95%,64%,${(0.5 * contrib).toFixed(3)})`);
          g.addColorStop(1, `hsla(${META_HUE},95%,64%,0)`);
          ctx.fillStyle = g;
          ctx.beginPath();
          ctx.arc(bx, by, pr * 2.4, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // ── Meta→ensemble edge (resolves with the verdict) ───────────────────
      ctx.lineWidth = lerp(1.2, 4.5, finalLit);
      ctx.strokeStyle = `hsla(${verdictHue},80%,58%,${(0.2 + 0.6 * eVerdict).toFixed(3)})`;
      ctx.beginPath();
      ctx.moveTo(mx + 26, my);
      ctx.lineTo(ex - 30, ey);
      ctx.stroke();

      // ── Base model nodes (left column) ───────────────────────────────────
      ctx.textBaseline = "middle";
      for (let i = 0; i < nb; i++) {
        const cx = px(baseX);
        const cy = py(nodeY[i]);
        const s = litBase[i];
        const r = 13;
        // glow when lit
        if (s > 0.04) {
          const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r + s * 16);
          g.addColorStop(0, `hsla(${verdictHue},92%,62%,${(0.42 * s).toFixed(3)})`);
          g.addColorStop(1, `hsla(${verdictHue},92%,62%,0)`);
          ctx.fillStyle = g;
          ctx.beginPath();
          ctx.arc(cx, cy, r + s * 16, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.fillStyle = `hsl(${verdictHue},${Math.round(lerp(10, 78, s))}%,${Math.round(
          lerp(26, 56, s),
        )}%)`;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = `hsla(${verdictHue},70%,72%,${(0.25 + 0.5 * s).toFixed(3)})`;
        ctx.stroke();

        // model name + its standalone metric, to the right of the node
        ctx.textAlign = "left";
        ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
        ctx.fillStyle = "rgba(214,224,238,0.92)";
        ctx.fillText(models[i].name, cx + r + 10, cy - 6);
        ctx.font = "500 10.5px ui-sans-serif, system-ui, sans-serif";
        ctx.fillStyle = "rgba(150,168,196,0.78)";
        ctx.fillText(
          `${data.meta.metricName} ${models[i].metric.toFixed(2)}`,
          cx + r + 10,
          cy + 8,
        );
        // live score inside the node
        ctx.textAlign = "center";
        ctx.font = "700 10px ui-sans-serif, system-ui, sans-serif";
        ctx.fillStyle = s > 0.04 ? "rgba(16,12,8,0.92)" : "rgba(190,204,226,0.55)";
        ctx.fillText((sample.baseScores[keys[i]] ?? 0).toFixed(2), cx, cy);
      }

      // ── Meta-learner node (centre) ───────────────────────────────────────
      const metaPulse = 0.5 + 0.5 * eFlow;
      const mr = 26;
      const mg = ctx.createRadialGradient(mx, my, 0, mx, my, mr + 18 * metaPulse);
      mg.addColorStop(0, `hsla(${META_HUE},95%,60%,${(0.4 * metaPulse).toFixed(3)})`);
      mg.addColorStop(1, `hsla(${META_HUE},95%,60%,0)`);
      ctx.fillStyle = mg;
      ctx.beginPath();
      ctx.arc(mx, my, mr + 18 * metaPulse, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = `hsl(${META_HUE},48%,${Math.round(lerp(30, 46, metaPulse))}%)`;
      ctx.beginPath();
      ctx.arc(mx, my, mr, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = `hsla(${META_HUE},85%,70%,0.7)`;
      ctx.stroke();
      ctx.textAlign = "center";
      ctx.font = "700 10px ui-sans-serif, system-ui, sans-serif";
      ctx.fillStyle = "rgba(248,240,224,0.95)";
      ctx.fillText("META", mx, my - 5);
      ctx.font = "600 9px ui-sans-serif, system-ui, sans-serif";
      ctx.fillStyle = "rgba(248,240,224,0.7)";
      ctx.fillText("stack", mx, my + 7);

      // ── Ensemble verdict node (right) ────────────────────────────────────
      const er = 30;
      const eg = ctx.createRadialGradient(ex, ey, 0, ex, ey, er + 22 * eVerdict);
      eg.addColorStop(0, `hsla(${verdictHue},92%,60%,${(0.5 * eVerdict).toFixed(3)})`);
      eg.addColorStop(1, `hsla(${verdictHue},92%,60%,0)`);
      ctx.fillStyle = eg;
      ctx.beginPath();
      ctx.arc(ex, ey, er + 22 * eVerdict, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = `hsl(${verdictHue},${Math.round(lerp(16, 72, eVerdict))}%,${Math.round(
        lerp(28, 52, eVerdict),
      )}%)`;
      ctx.beginPath();
      ctx.arc(ex, ey, er, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = `hsla(${verdictHue},80%,72%,${(0.3 + 0.5 * eVerdict).toFixed(3)})`;
      ctx.stroke();
      ctx.fillStyle = "rgba(248,244,238,0.96)";
      ctx.font = "800 16px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText(finalLit.toFixed(2), ex, ey - 4);
      ctx.font = "700 9px ui-sans-serif, system-ui, sans-serif";
      ctx.fillStyle = "rgba(248,244,238,0.82)";
      if (eVerdict > 0.55) {
        ctx.fillText(fraud ? "FRAUD" : "LEGIT", ex, ey + 11);
      } else {
        ctx.fillText("…", ex, ey + 11);
      }
      ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
      ctx.fillStyle = "rgba(176,192,214,0.62)";
      ctx.fillText("ENSEMBLE", ex, ey + er + 14);

      // ── Throttled UI sync ────────────────────────────────────────────────
      const uiSample = scrub !== null ? scrub : fromSample;
      if (uiSample !== lastUiSample) {
        lastUiSample = uiSample;
        setActiveSample(uiSample);
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
        if (visible) start();
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
    // Re-create loop when data/precomp ready or pause toggles.
  }, [data, precomp, paused]);

  const activeSampleData = data?.samples[activeSample];
  const caption = activeSampleData
    ? activeSampleData.label === 1
      ? "Five base models vote — the meta-learner weighs each and resolves a FRAUD verdict."
      : "Five base models vote — the meta-learner weighs each and clears this as LEGIT."
    : "";

  // ── Fallback ──────────────────────────────────────────────────────────────
  if (error) {
    return (
      <section
        aria-label="Stacked federated ensemble"
        className="rise overflow-hidden rounded-[20px] border bg-bg-surface/80 p-6 backdrop-blur-sm sm:p-8"
        style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-panel)" }}
      >
        <p className="eyebrow text-accent-gold">Stacked federated ensemble</p>
        <h2 className="mt-2 font-display text-[clamp(1.5rem,1.1rem+1.6vw,2.4rem)] leading-tight tracking-tight text-text-primary">
          Five models, one verdict — better than any alone.
        </h2>
        <p className="mt-4 text-[13px] leading-relaxed text-text-secondary">
          Ensemble telemetry is unavailable right now. The stacked model still combines a logistic,
          MLP, federated GBDT, GRU sequence and embedding base learner via a federated meta-learner —
          this visualization will resume once the data feed returns.
        </p>
      </section>
    );
  }

  return (
    <section
      aria-labelledby="ensemble-stack-heading"
      className="rise overflow-hidden rounded-[20px] border bg-bg-surface/80 backdrop-blur-sm"
      style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-panel)" }}
    >
      <header
        className="grid gap-4 border-b p-5 sm:p-6 lg:grid-cols-[1fr_auto] lg:items-end"
        style={{ borderColor: "var(--border-default)" }}
      >
        <div>
          <p className="eyebrow text-accent-gold">Stacked federated ensemble</p>
          <h2
            id="ensemble-stack-heading"
            className="mt-2 font-display text-[clamp(1.7rem,1.2rem+2vw,3rem)] leading-tight tracking-tight text-text-primary"
          >
            Five models, one verdict — better than any alone.
          </h2>
          <p className="mt-3 max-w-3xl text-[13px] leading-relaxed text-text-secondary sm:text-[14px]">
            Logistic, MLP, federated GBDT, a GRU sequence model and categorical embeddings each
            catch a different fraud typology — and miss the rest. A federated logistic meta-learner
            stacks their votes, trusting each by its learned weight, to beat every single model.
          </p>
        </div>
        <div
          className="grid grid-cols-2 gap-px overflow-hidden rounded-[14px] border text-center sm:min-w-[300px]"
          style={{ borderColor: "var(--border-default)", background: "var(--border-default)" }}
        >
          <Stat
            label="best single"
            value={bestSingle.toFixed(2)}
            sub={data?.meta.metricName ?? "recall"}
            tone="gold"
          />
          <Stat
            label="ensemble"
            value={(data?.ensemble.metric ?? 0).toFixed(2)}
            sub={data?.meta.metricName ?? "recall"}
            tone="fed"
          />
        </div>
      </header>

      <div className="grid gap-4 p-4 sm:p-5 lg:grid-cols-[1.45fr_1fr]">
        {/* ── Animated stack canvas ── */}
        <div
          ref={wrapRef}
          className="relative h-[360px] w-full overflow-hidden rounded-[14px] border bg-bg-deep sm:h-[420px]"
          style={{ borderColor: "var(--border-default)" }}
        >
          <canvas
            ref={canvasRef}
            className="block h-full w-full"
            role="img"
            aria-label="Five base models feeding a meta-learner that outputs the ensemble fraud verdict"
          />
          <p
            className="pointer-events-none absolute bottom-3 left-3 right-3 text-[12px] leading-snug text-text-secondary sm:text-[13px]"
            style={{ textShadow: "0 1px 8px rgba(8,12,20,0.92)" }}
          >
            {caption}
          </p>
        </div>

        {/* ── Per-model metric vs ensemble comparison bars ── */}
        <div
          className="flex flex-col gap-2.5 rounded-[14px] border p-4"
          style={{ borderColor: "var(--border-default)", background: "var(--bg-surface-2)" }}
        >
          <p className="eyebrow text-text-muted">
            {data?.meta.metricName ?? "recall"} · each model vs the stack
          </p>
          {data?.models.map((m) => (
            <MetricBar
              key={m.key}
              name={m.name}
              metric={m.metric}
              weight={m.weight}
              max={data.ensemble.metric}
            />
          ))}
          <MetricBar
            name="Ensemble"
            metric={data?.ensemble.metric ?? 0}
            max={data?.ensemble.metric ?? 1}
            isEnsemble
          />
        </div>
      </div>

      {/* ── Controls: play/pause + sample scrubber ── */}
      <div
        className="flex flex-wrap items-center gap-3 border-t px-4 py-3 sm:px-5"
        style={{ borderColor: "var(--border-default)" }}
      >
        <button
          type="button"
          onClick={() => setPaused((p) => !p)}
          className="rounded-full border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-text-secondary transition-colors hover:text-text-primary"
          style={{ borderColor: "var(--border-strong)", background: "var(--bg-surface-2)" }}
        >
          {paused ? "play" : "pause"}
        </button>

        <div className="flex items-center gap-1.5">
          {Array.from({ length: sampleCount }).map((_, i) => {
            const active = i === activeSample;
            return (
              <button
                key={i}
                type="button"
                aria-label={`Go to transaction ${i + 1}`}
                aria-pressed={active}
                onClick={() => {
                  scrubRef.current = i;
                  setScrubbing(true);
                  setActiveSample(i);
                  setPaused(true);
                }}
                className="h-2 rounded-full transition-all"
                style={{
                  width: active ? 22 : 10,
                  background: active ? "var(--accent-gold)" : "var(--border-strong)",
                }}
              />
            );
          })}
        </div>

        {scrubbing && (
          <button
            type="button"
            onClick={() => {
              scrubRef.current = null;
              setScrubbing(false);
              setPaused(false);
            }}
            className="rounded-full border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-text-muted transition-colors hover:text-text-primary"
            style={{ borderColor: "var(--border-strong)", background: "var(--bg-surface-2)" }}
          >
            resume auto
          </button>
        )}

        <p className="tabular ml-auto text-[12px] text-text-muted">
          transaction {activeSample + 1}/{sampleCount} ·{" "}
          {activeSampleData?.label === 1 ? "true fraud" : "true legit"}
        </p>
      </div>
    </section>
  );
}

function Stat({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: string;
  sub: string;
  tone: "fed" | "gold";
}) {
  const color = tone === "fed" ? "var(--fed)" : "var(--accent-gold)";
  return (
    <div className="bg-bg-surface-2 px-4 py-3">
      <p className="eyebrow" style={{ color }}>
        {label}
      </p>
      <p className="tabular mt-1 font-display text-2xl leading-none text-text-primary">{value}</p>
      <p className="mt-1 text-[10px] uppercase tracking-[0.12em] text-text-muted">{sub}</p>
    </div>
  );
}

function MetricBar({
  name,
  metric,
  weight,
  max,
  isEnsemble,
}: {
  name: string;
  metric: number;
  weight?: number;
  max: number;
  isEnsemble?: boolean;
}) {
  const pct = Math.max(0, Math.min(1, metric / Math.max(1e-6, max))) * 100;
  const color = isEnsemble ? "var(--fed)" : "var(--accent-gold)";
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between text-[11px]">
        <span
          className={isEnsemble ? "font-semibold text-text-primary" : "text-text-secondary"}
        >
          {name}
        </span>
        <span className="tabular flex items-center gap-2 text-text-muted">
          {weight !== undefined && (
            <span
              className="rounded px-1 py-px text-[9px]"
              style={{
                background: weight >= 0 ? "rgba(214,177,98,0.14)" : "rgba(120,140,170,0.14)",
                color: weight >= 0 ? "var(--accent-gold)" : "var(--text-muted)",
              }}
              title="meta-learner weight (how much the stack trusts this model)"
            >
              w {weight >= 0 ? "+" : ""}
              {weight.toFixed(1)}
            </span>
          )}
          {metric.toFixed(2)}
        </span>
      </div>
      <div
        className="h-2 w-full overflow-hidden rounded-full"
        style={{ background: "var(--border-default)" }}
      >
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

const EnsembleStack = memo(EnsembleStackInner);
export default EnsembleStack;
