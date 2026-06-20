"use client";

import { memo, useEffect, useMemo, useRef, useState } from "react";

// ── Data schema (web/public/viz/graph.json) ────────────────────────────────
type GraphNode = {
  id: number;
  bank: number; // 0..7
  x: number; // [-1, 1] precomputed force-directed layout
  y: number; // [-1, 1]
  mule: boolean;
};

type GraphFrame = {
  round: number;
  scores: number[]; // 0..1 per node, aligned with nodes[] index
};

type GraphData = {
  nodes: GraphNode[];
  edges: [number, number][]; // node-index pairs
  frames: GraphFrame[];
  meta: { banks: number; note: string };
};

// Per-round caption — narrative arc from blind silo to federated reveal.
const CAPTIONS = [
  "Round 0 — no single bank sees the whole ring.",
  "Round 1 — local signal stirs on a few accounts.",
  "Round 2 — neighbours start exchanging suspicion.",
  "Round 4 — message-passing links the corridors.",
  "Round 6 — the cross-bank ring begins to glow.",
  "Round 9 — federation surfaces the cross-bank mule network.",
] as const;

// Tuning ──────────────────────────────────────────────────────────────────
const STEP_MS = 1200; // time per round transition
const FINAL_PAUSE_MS = 2200; // extra dwell on the last frame before looping
const HOT_THRESHOLD = 0.5; // score above which an account is "surfaced"

// Colour ramp between calm slate-teal and hot amber/rose. Returns rgb tuple.
function rampColor(score: number, out: [number, number, number]) {
  // calm: slate-teal #3f6f7a-ish → warm: rose #fb7185 → hot: amber #fbbf24
  const s = clamp01(score);
  if (s < 0.5) {
    // slate-teal → rose
    const t = s / 0.5;
    out[0] = lerp(63, 251, t);
    out[1] = lerp(111, 113, t);
    out[2] = lerp(122, 133, t);
  } else {
    // rose → amber
    const t = (s - 0.5) / 0.5;
    out[0] = lerp(251, 251, t);
    out[1] = lerp(113, 191, t);
    out[2] = lerp(133, 36, t);
  }
  return out;
}

function clamp01(v: number) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}
function easeInOut(t: number) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

function MuleGraphInner() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  const [data, setData] = useState<GraphData | null>(null);
  const [error, setError] = useState(false);

  // UI-facing state, updated at a throttled cadence from the RAF loop.
  const [activeRound, setActiveRound] = useState(0);
  const [hotCount, setHotCount] = useState(0);
  const [paused, setPaused] = useState(false);

  // Manual scrub override: when set, the loop holds on this frame index.
  const scrubRef = useRef<number | null>(null);

  // ── Fetch own data ───────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    fetch("/viz/graph.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((json: GraphData) => {
        if (cancelled) return;
        if (!json?.nodes?.length || !json?.frames?.length) throw new Error("empty");
        setData(json);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Total mules — denominator for the recall stat (computed once).
  const totalMules = useMemo(
    () => (data ? data.nodes.reduce((n, node) => n + (node.mule ? 1 : 0), 0) : 0),
    [data],
  );
  const frameCount = data?.frames.length ?? 0;

  // ── Precompute edge endpoints + cross-bank flags (no per-frame allocation) ─
  // Stored as flat arrays in screen-normalised [0,1] space derived from x,y.
  const precomp = useMemo(() => {
    if (!data) return null;
    const { nodes, edges } = data;
    const nx = new Float32Array(nodes.length); // base normalised x [0,1]
    const ny = new Float32Array(nodes.length);
    for (let i = 0; i < nodes.length; i++) {
      nx[i] = (nodes[i].x + 1) / 2;
      ny[i] = (nodes[i].y + 1) / 2;
    }
    const ea = new Int32Array(edges.length); // endpoint a index
    const eb = new Int32Array(edges.length); // endpoint b index
    const cross = new Uint8Array(edges.length); // 1 if endpoints differ in bank
    let crossCount = 0;
    for (let e = 0; e < edges.length; e++) {
      const a = edges[e][0];
      const b = edges[e][1];
      ea[e] = a;
      eb[e] = b;
      const c = nodes[a].bank !== nodes[b].bank ? 1 : 0;
      cross[e] = c;
      crossCount += c;
    }
    // Reusable score buffer interpolated each draw — allocated once.
    const interp = new Float32Array(nodes.length);
    return { nx, ny, ea, eb, cross, crossCount, interp };
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

    const { nx, ny, ea, eb, cross, interp } = precomp;
    const { nodes, frames } = data;
    const n = nodes.length;
    const lastFrame = frames.length - 1;

    let width = 0;
    let height = 0;
    let dpr = 1;

    const sizeCanvas = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = wrap.getBoundingClientRect();
      width = Math.max(320, Math.floor(rect.width));
      height = Math.max(320, Math.floor(rect.height));
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    sizeCanvas();

    // Loop bookkeeping — all in closure-local mutable state, not React state.
    // The effect re-runs whenever `paused` flips, so a plain capture is correct.
    let raf = 0;
    let visible = true;
    const pausedLocal = paused;
    const startTs = performance.now();
    // Position within the timeline: which step we're transitioning across.
    let stepStart = performance.now();
    let fromFrame = 0;

    let lastUiRound = -1;
    let lastUiHot = -1;

    const rgb: [number, number, number] = [0, 0, 0];

    // Maps a normalised [0,1] coord to padded device-independent pixels.
    const PAD = 26;
    const px = (v: number) => PAD + v * (width - PAD * 2);
    const py = (v: number) => PAD + v * (height - PAD * 2);

    const draw = (now: number) => {
      // ── Resolve the interpolation target ──────────────────────────────────
      const scrub = scrubRef.current;
      let toFrame: number;
      let phase: number; // 0..1 eased blend between fromFrame → toFrame

      if (scrub !== null) {
        // Manual hold on a scrubbed frame — snap, no auto-advance.
        fromFrame = scrub;
        toFrame = scrub;
        phase = 1;
        stepStart = now;
      } else if (pausedLocal || reduceMotion) {
        toFrame = fromFrame;
        phase = 1;
        stepStart = now;
      } else {
        const dwell = fromFrame === lastFrame ? STEP_MS + FINAL_PAUSE_MS : STEP_MS;
        const raw = (now - stepStart) / dwell;
        if (raw >= 1) {
          // advance
          fromFrame = fromFrame >= lastFrame ? 0 : fromFrame + 1;
          stepStart = now;
          toFrame = fromFrame === lastFrame ? fromFrame : fromFrame + 1;
          phase = 0;
        } else {
          toFrame = fromFrame >= lastFrame ? fromFrame : fromFrame + 1;
          phase = easeInOut(clamp01(raw));
        }
      }

      const fa = frames[fromFrame].scores;
      const fb = frames[toFrame].scores;

      // Interpolate per-node scores + tally hot accounts in one pass.
      let hot = 0;
      for (let i = 0; i < n; i++) {
        const s = fa[i] + (fb[i] - fa[i]) * phase;
        interp[i] = s;
        if (nodes[i].mule && s > HOT_THRESHOLD) hot++;
      }

      // Gentle idle breathing (purely cosmetic positional drift).
      const breath = reduceMotion ? 0 : Math.sin((now - startTs) / 2600) * 0.5;
      const drift = reduceMotion ? 0 : 1.4;

      // ── Background ────────────────────────────────────────────────────────
      ctx.clearRect(0, 0, width, height);
      const bg = ctx.createLinearGradient(0, 0, width, height);
      bg.addColorStop(0, "rgba(10,14,22,0.96)");
      bg.addColorStop(0.6, "rgba(13,19,32,0.96)");
      bg.addColorStop(1, "rgba(19,27,44,0.96)");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, width, height);

      // ── Edges: same-bank first (subtle), then cross-bank (accented) ───────
      // Two passes so each strokeStyle is set once.
      ctx.lineWidth = 0.7;
      ctx.strokeStyle = "rgba(110,122,144,0.13)";
      ctx.beginPath();
      for (let e = 0; e < ea.length; e++) {
        if (cross[e]) continue;
        const a = ea[e];
        const b = eb[e];
        ctx.moveTo(px(nx[a]) + dx(a), py(ny[a]) + dy(a));
        ctx.lineTo(px(nx[b]) + dx(b), py(ny[b]) + dy(b));
      }
      ctx.stroke();

      // Cross-bank links accented — brightness keyed to the hotter endpoint.
      for (let e = 0; e < ea.length; e++) {
        if (!cross[e]) continue;
        const a = ea[e];
        const b = eb[e];
        const heat = Math.max(interp[a], interp[b]);
        const alpha = 0.16 + heat * 0.42;
        ctx.strokeStyle = `rgba(214,168,91,${alpha.toFixed(3)})`;
        ctx.lineWidth = 0.8 + heat * 1.1;
        ctx.beginPath();
        ctx.moveTo(px(nx[a]) + dx(a), py(ny[a]) + dy(a));
        ctx.lineTo(px(nx[b]) + dx(b), py(ny[b]) + dy(b));
        ctx.stroke();
      }

      // ── Glow pass (additive) for hot nodes ────────────────────────────────
      ctx.globalCompositeOperation = "lighter";
      for (let i = 0; i < n; i++) {
        const s = interp[i];
        if (s < 0.34) continue;
        const cx = px(nx[i]) + dx(i);
        const cy = py(ny[i]) + dy(i);
        rampColor(s, rgb);
        const glowR = 6 + s * 22;
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, glowR);
        const a = (s - 0.3) * 0.5;
        g.addColorStop(0, `rgba(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0},${a.toFixed(3)})`);
        g.addColorStop(1, `rgba(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0},0)`);
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(cx, cy, glowR, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalCompositeOperation = "source-over";

      // ── Node bodies ───────────────────────────────────────────────────────
      for (let i = 0; i < n; i++) {
        const s = interp[i];
        const cx = px(nx[i]) + dx(i);
        const cy = py(ny[i]) + dy(i);
        const r = 2.4 + s * 4.2;
        rampColor(s, rgb);
        ctx.fillStyle = `rgb(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0})`;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
        // Outline mule nodes so the ground-truth ring is legible.
        if (nodes[i].mule) {
          ctx.lineWidth = 1;
          ctx.strokeStyle = `rgba(242,239,230,${(0.28 + s * 0.55).toFixed(3)})`;
          ctx.stroke();
        }
      }

      // ── Throttled UI sync ────────────────────────────────────────────────
      const displayRound = frames[scrub !== null ? scrub : toFrame].round;
      if (displayRound !== lastUiRound) {
        lastUiRound = displayRound;
        setActiveRound(scrub !== null ? scrub : toFrame);
      }
      if (hot !== lastUiHot) {
        lastUiHot = hot;
        setHotCount(hot);
      }

      function dx(i: number) {
        return drift * Math.sin((nx[i] + ny[i]) * 6 + breath);
      }
      function dy(i: number) {
        return drift * Math.cos((nx[i] - ny[i]) * 6 + breath);
      }

      if (visible) raf = requestAnimationFrame(draw);
    };

    const start = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(draw);
    };
    const stop = () => cancelAnimationFrame(raf);

    // Pause offscreen.
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
    // Re-create loop when data/precomp ready or pause toggles.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, precomp, paused]);

  const caption =
    CAPTIONS[Math.min(activeRound, CAPTIONS.length - 1)] ?? CAPTIONS[CAPTIONS.length - 1];
  const recallPct = totalMules > 0 ? Math.round((hotCount / totalMules) * 100) : 0;
  const roundLabel =
    data && data.frames[activeRound] ? data.frames[activeRound].round : activeRound;

  // ── Fallback ────────────────────────────────────────────────────────────
  if (error) {
    return (
      <section
        aria-label="Cross-bank mule graph"
        className="rise overflow-hidden rounded-[20px] border bg-bg-surface/80 p-6 backdrop-blur-sm sm:p-8"
        style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-panel)" }}
      >
        <p className="eyebrow text-accent-gold">Federated GNN · cross-bank mule graph</p>
        <h2 className="mt-2 font-display text-[clamp(1.5rem,1.1rem+1.6vw,2.4rem)] leading-tight tracking-tight text-text-primary">
          The ring no single bank could see.
        </h2>
        <p className="mt-4 text-[13px] leading-relaxed text-text-secondary">
          Graph telemetry is unavailable right now. The federated GraphSAGE model still propagates
          fraud scores across institutions — this visualization will resume once the data feed
          returns.
        </p>
      </section>
    );
  }

  return (
    <section
      aria-labelledby="mule-graph-heading"
      className="rise overflow-hidden rounded-[20px] border bg-bg-surface/80 backdrop-blur-sm"
      style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-panel)" }}
    >
      <header
        className="grid gap-4 border-b p-5 sm:p-6 lg:grid-cols-[1fr_auto] lg:items-end"
        style={{ borderColor: "var(--border-default)" }}
      >
        <div>
          <p className="eyebrow text-accent-gold">Federated GNN · cross-bank mule graph</p>
          <h2
            id="mule-graph-heading"
            className="mt-2 font-display text-[clamp(1.7rem,1.2rem+2vw,3rem)] leading-tight tracking-tight text-text-primary"
          >
            The ring no single bank could see.
          </h2>
          <p className="mt-3 max-w-3xl text-[13px] leading-relaxed text-text-secondary sm:text-[14px]">
            A federated GraphSAGE GNN passes messages across {data?.meta.banks ?? 8} institutions
            without moving a single record. Watch fraud scores propagate over rounds until the
            cross-bank mule ring lights up.
          </p>
        </div>
        <div
          className="grid grid-cols-2 gap-px overflow-hidden rounded-[14px] border text-center sm:min-w-[300px]"
          style={{ borderColor: "var(--border-default)", background: "var(--border-default)" }}
        >
          <Stat label="round" value={String(roundLabel).padStart(2, "0")} tone="gold" />
          <Stat label="mules surfaced" value={`${recallPct}%`} tone="fed" />
        </div>
      </header>

      <div className="p-4 sm:p-5">
        <div
          ref={wrapRef}
          className="relative h-[360px] w-full overflow-hidden rounded-[14px] border bg-bg-deep sm:h-[440px]"
          style={{ borderColor: "var(--border-default)" }}
        >
          <canvas
            ref={canvasRef}
            className="block h-full w-full"
            role="img"
            aria-label="Cross-bank mule ring graph with fraud scores propagating over rounds"
          />

          {/* Legend overlay */}
          <div
            className="pointer-events-none absolute left-3 top-3 flex flex-col gap-1.5 rounded-[10px] border px-3 py-2 text-[11px] backdrop-blur-sm"
            style={{
              borderColor: "var(--border-default)",
              background: "rgba(13,19,32,0.66)",
            }}
          >
            <LegendRow color="#fbbf24" ring label="mule ring" />
            <LegendRow color="#3f6f7a" label="normal account" />
            <LegendRow color="#d6a85b" line label="cross-bank link" />
          </div>

          {/* Caption overlay */}
          <p
            className="pointer-events-none absolute bottom-3 left-3 right-3 text-[12px] leading-snug text-text-secondary sm:text-[13px]"
            style={{ textShadow: "0 1px 8px rgba(10,14,22,0.9)" }}
          >
            {caption}
          </p>
        </div>

        {/* Scrubber + controls */}
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
            {Array.from({ length: frameCount }).map((_, i) => {
              const active = i === activeRound;
              return (
                <button
                  key={i}
                  type="button"
                  aria-label={`Go to round ${data?.frames[i]?.round ?? i}`}
                  aria-pressed={active}
                  onClick={() => {
                    scrubRef.current = i;
                    setActiveRound(i);
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

          {scrubRef.current !== null && (
            <button
              type="button"
              onClick={() => {
                scrubRef.current = null;
                setPaused(false);
              }}
              className="rounded-full border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-text-muted transition-colors hover:text-text-primary"
              style={{ borderColor: "var(--border-strong)", background: "var(--bg-surface-2)" }}
            >
              resume auto
            </button>
          )}

          <p className="tabular ml-auto text-[12px] text-text-muted">
            {hotCount}/{totalMules} mule accounts surfaced
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

function LegendRow({
  color,
  label,
  ring,
  line,
}: {
  color: string;
  label: string;
  ring?: boolean;
  line?: boolean;
}) {
  return (
    <span className="flex items-center gap-2 text-text-secondary">
      {line ? (
        <span
          aria-hidden
          className="inline-block h-[2px] w-4 rounded-full"
          style={{ background: color }}
        />
      ) : (
        <span
          aria-hidden
          className="inline-block h-2.5 w-2.5 rounded-full"
          style={{
            background: color,
            boxShadow: ring ? "0 0 6px 1px rgba(251,191,36,0.6)" : undefined,
            border: ring ? "1px solid rgba(242,239,230,0.7)" : undefined,
          }}
        />
      )}
      {label}
    </span>
  );
}

const MuleGraph = memo(MuleGraphInner);
export default MuleGraph;
