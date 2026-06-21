"use client";
import { memo, useEffect, useMemo, useRef, useState } from "react";

/*
  FederationPulse — visualizes ONE federated aggregation round.
  Four beats: arrive -> clip -> select -> aggregate, auto-advancing then looping.

  Data: /viz/federation.json (PCA-2D positions of 8 client model-updates).
  One poisoned client arrives far outside the DP clip sphere; it is DP-clipped
  onto the boundary, then Multi-Krum rejects it; the rest aggregate into a new
  global point. Self-contained client component, no required props.
*/

type Client = {
  xy: [number, number];
  poisoned: boolean;
  rejected: boolean;
  norm: number;
};

type Frame = {
  stage: "arrive" | "clip" | "select" | "aggregate";
  clients: Client[];
  clipRadius: number;
  global: [number, number];
};

type FederationData = {
  frames: Frame[];
  meta: { krum: string; rejected: number; note: string };
};

const STAGES = [
  { key: "arrive", label: "Arrive" },
  { key: "clip", label: "DP clip" },
  { key: "select", label: "Multi-Krum" },
  { key: "aggregate", label: "Aggregate" },
] as const;

const CAPTIONS: Record<Frame["stage"], string> = {
  arrive: "Eight client model-updates arrive — one poisoned update sits far outside the bound.",
  clip: "Differential-privacy clipping bounds every update onto the clip sphere.",
  select: "Multi-Krum drops the Byzantine outlier.",
  aggregate:
    "The rest aggregate into the new global model — no raw data ever moved.",
};

const BEAT_MS = 1600; // dwell per beat
const TWEEN_MS = 700; // transition into a beat
const LOOP_PAUSE_MS = 900; // extra pause before looping back to beat 0

// Theme tokens (mirror globals.css). RGB triples so we can vary alpha cheaply.
const TEAL = "52, 211, 153";
const ROSE = "240, 74, 82";
const GOLD = "214, 168, 91";
const TEXT = "242, 239, 230";
const MUTED = "111, 122, 144";

// World->screen scale: map ~1.6 PCA units to a comfortable canvas radius so the
// clipRadius ring reads clearly and the far poison (x~1.44) stays on-canvas.
const WORLD_SPAN = 1.7;

function easeOut(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}
function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function FederationPulse() {
  const [data, setData] = useState<FederationData | null>(null);
  const [error, setError] = useState(false);
  const [beat, setBeat] = useState(0); // active frame index 0..3
  const [paused, setPaused] = useState(false); // hovered / offscreen pause

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Animation state kept in refs so the RAF loop is never recreated per render.
  const beatRef = useRef(0); // current target beat (mirrors `beat` state)
  const prevBeatRef = useRef(0); // beat we are tweening FROM
  const tweenStartRef = useRef(0); // performance.now() at last beat change
  const autoPausedRef = useRef(false); // offscreen / hover
  const visibleRef = useRef(true);

  // Per-client fly-in entry offsets (stable across frames; computed once on load).
  const entryRef = useRef<Array<{ ex: number; ey: number }>>([]);

  // Scratch points reused every frame — no per-frame allocation.
  const scratchRef = useRef<Array<{ x: number; y: number }>>([]);

  useEffect(() => {
    let cancelled = false;
    fetch("viz/federation.json")
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status));
        return r.json();
      })
      .then((json: FederationData) => {
        if (cancelled) return;
        if (!json?.frames?.length) throw new Error("empty");
        // Deterministic-ish entry vectors fanning out from the canvas edges.
        const n = json.frames[0].clients.length;
        const entries: Array<{ ex: number; ey: number }> = [];
        for (let i = 0; i < n; i++) {
          const a = (i / n) * Math.PI * 2 + 0.6;
          entries.push({ ex: Math.cos(a) * 2.4, ey: Math.sin(a) * 2.4 });
        }
        entryRef.current = entries;
        scratchRef.current = Array.from({ length: n }, () => ({ x: 0, y: 0 }));
        setData(json);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Keep refs in sync with the beat state and arm the tween clock on each change.
  useEffect(() => {
    if (beat !== beatRef.current) {
      prevBeatRef.current = beatRef.current;
      beatRef.current = beat;
      tweenStartRef.current = performance.now();
    }
  }, [beat]);

  // Auto-advance the beats. A single interval that respects pause state.
  useEffect(() => {
    if (!data || paused) return;
    const frames = data.frames.length;
    const isLoopEdge = beat === frames - 1;
    const id = window.setTimeout(
      () => setBeat((b) => (b + 1) % frames),
      BEAT_MS + (isLoopEdge ? LOOP_PAUSE_MS : 0),
    );
    return () => window.clearTimeout(id);
  }, [data, beat, paused]);

  // Pause when offscreen.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        const vis = entries[0]?.isIntersecting ?? true;
        visibleRef.current = vis;
        autoPausedRef.current = !vis;
        setPaused((prev) => {
          const next = !vis || hoverPausedRef.current;
          return next === prev ? prev : next;
        });
      },
      { threshold: 0.15 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const hoverPausedRef = useRef(false);

  // The RAF render loop — created once, reads everything from refs.
  useEffect(() => {
    if (!data) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    let width = 0;
    let height = 0;
    const sizeCanvas = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      width = Math.max(320, Math.floor(rect.width));
      height = Math.max(300, Math.floor(rect.height));
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    sizeCanvas();

    const frames = data.frames;
    let raf = 0;

    const render = (now: number) => {
      const cur = beatRef.current;
      const prev = prevBeatRef.current;
      const rawT = reduceMotion
        ? 1
        : Math.min(1, (now - tweenStartRef.current) / TWEEN_MS);
      const t = easeInOut(rawT);

      drawScene(
        ctx,
        frames,
        prev,
        cur,
        t,
        now,
        width,
        height,
        entryRef.current,
        scratchRef.current,
        reduceMotion,
      );

      // Always keep painting (cheap; 8 dots) so glows/pulses stay alive; pausing
      // only freezes beat advance, not the canvas itself.
      raf = requestAnimationFrame(render);
    };
    raf = requestAnimationFrame(render);

    let first = true;
    const ro = new ResizeObserver(() => {
      if (first) {
        first = false;
        return;
      }
      sizeCanvas();
    });
    ro.observe(canvas);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [data]);

  const meta = data?.meta;
  const arriveFrame = data?.frames?.[0];
  const poisonNorm = useMemo(() => {
    const p = arriveFrame?.clients.find((c) => c.poisoned);
    return p ? p.norm : null;
  }, [arriveFrame]);

  const onEnter = () => {
    hoverPausedRef.current = true;
    setPaused(true);
  };
  const onLeave = () => {
    hoverPausedRef.current = false;
    if (visibleRef.current) setPaused(false);
  };

  const goPrev = () => {
    if (!data) return;
    setPaused(true);
    hoverPausedRef.current = true;
    setBeat((b) => (b - 1 + data.frames.length) % data.frames.length);
  };
  const goNext = () => {
    if (!data) return;
    setPaused(true);
    hoverPausedRef.current = true;
    setBeat((b) => (b + 1) % data.frames.length);
  };

  return (
    <section
      ref={containerRef}
      aria-labelledby="federation-pulse-heading"
      className="rise overflow-hidden rounded-[20px] border bg-bg-surface/80 backdrop-blur-sm"
      style={{
        borderColor: "var(--border-default)",
        boxShadow: "var(--shadow-panel)",
      }}
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
    >
      <header
        className="grid gap-4 border-b p-5 sm:p-6 lg:grid-cols-[1fr_auto] lg:items-end"
        style={{ borderColor: "var(--border-default)" }}
      >
        <div>
          <p className="eyebrow text-accent-gold">
            Secure aggregation · DP + Multi-Krum
          </p>
          <h2
            id="federation-pulse-heading"
            className="mt-2 font-display text-[clamp(1.7rem,1.2rem+2vw,3rem)] leading-tight tracking-tight text-text-primary"
          >
            Poison in, clean model out.
          </h2>
          <p className="mt-3 max-w-2xl text-[13px] leading-relaxed text-text-secondary sm:text-[14px]">
            One federated round: eight banks each submit a privacy-clipped model
            update. {meta?.krum === "multi-krum" ? "Multi-Krum" : "Krum"} rejects
            the single poisoned contribution before the rest aggregate — without
            any raw record ever leaving a bank.
          </p>
        </div>
        <div
          className="grid grid-cols-3 gap-px overflow-hidden rounded-[14px] border text-center sm:min-w-[360px]"
          style={{
            borderColor: "var(--border-default)",
            background: "var(--border-default)",
          }}
        >
          <Metric label="Clients" value="8" tone="text" />
          <Metric
            label="Rejected"
            value={meta ? String(meta.rejected) : "—"}
            tone="silo"
          />
          <Metric
            label="Poison ‖Δ‖"
            value={poisonNorm ? poisonNorm.toFixed(1) : "—"}
            tone="silo"
          />
        </div>
      </header>

      {/* Stepper */}
      <div
        className="flex flex-wrap items-center gap-2 border-b px-5 py-3 sm:px-6"
        style={{ borderColor: "var(--border-default)" }}
      >
        {STAGES.map((stage, i) => {
          const active = i === beat;
          const done = i < beat;
          const color = active
            ? "var(--accent-gold)"
            : done
              ? "var(--fed)"
              : "var(--text-muted)";
          return (
            <button
              key={stage.key}
              type="button"
              onClick={() => {
                if (!data) return;
                hoverPausedRef.current = true;
                setPaused(true);
                setBeat(i);
              }}
              aria-current={active ? "step" : undefined}
              className="inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.12em] transition-colors"
              style={{
                borderColor: active
                  ? "rgba(214,168,91,0.45)"
                  : "var(--border-strong)",
                background: active
                  ? "rgba(214,168,91,0.08)"
                  : "var(--bg-surface-2)",
                color,
              }}
            >
              <span
                aria-hidden
                className="tabular grid h-4 w-4 place-items-center rounded-full text-[9px]"
                style={{
                  background: active
                    ? "var(--accent-gold)"
                    : "var(--bg-surface)",
                  color: active ? "var(--bg-deep)" : color,
                }}
              >
                {i + 1}
              </span>
              {stage.label}
            </button>
          );
        })}
      </div>

      <div className="p-4 sm:p-5">
        {error ? (
          <Fallback />
        ) : (
          <>
            <canvas
              ref={canvasRef}
              className="block h-[340px] w-full rounded-[14px] border bg-bg-deep sm:h-[420px]"
              style={{ borderColor: "var(--border-default)" }}
              role="img"
              aria-label={`Federated aggregation round, ${STAGES[beat].label} stage`}
            />

            {/* Caption + transport */}
            <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <p
                key={beat}
                className="rise max-w-2xl text-[13px] leading-relaxed text-text-secondary sm:text-[14px]"
              >
                <span
                  className="eyebrow mr-2 align-middle"
                  style={{ color: "var(--accent-gold)" }}
                >
                  Beat {beat + 1}
                </span>
                {CAPTIONS[STAGES[beat].key]}
              </p>
              <div className="flex shrink-0 items-center gap-2">
                <TransportButton label="Prev beat" onClick={goPrev}>
                  ‹ Prev
                </TransportButton>
                <TransportButton label="Next beat" onClick={goNext}>
                  Next ›
                </TransportButton>
              </div>
            </div>

            <Legend />
          </>
        )}
      </div>
    </section>
  );
}

function drawScene(
  ctx: CanvasRenderingContext2D,
  frames: Frame[],
  prevBeat: number,
  curBeat: number,
  t: number,
  now: number,
  width: number,
  height: number,
  entries: Array<{ ex: number; ey: number }>,
  scratch: Array<{ x: number; y: number }>,
  reduceMotion: boolean,
) {
  const prev = frames[prevBeat];
  const cur = frames[curBeat];

  // Background.
  ctx.clearRect(0, 0, width, height);
  const bg = ctx.createLinearGradient(0, 0, width, height);
  bg.addColorStop(0, "rgba(10,14,22,0.98)");
  bg.addColorStop(0.6, "rgba(13,19,32,0.98)");
  bg.addColorStop(1, "rgba(13,110,79,0.10)");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  const cx = width / 2;
  const cy = height / 2;
  const scale = (Math.min(width, height) * 0.5) / WORLD_SPAN;
  const toX = (wx: number) => cx + wx * scale;
  const toY = (wy: number) => cy - wy * scale;

  drawGrid(ctx, cx, cy, width, height);

  const stage = cur.stage;
  const clipR = lerp(prev.clipRadius, cur.clipRadius, t) * scale;

  // The DP clip ring appears from the "clip" stage onward.
  const clipStageIdx = 1;
  const showClip = curBeat >= clipStageIdx;
  if (showClip) {
    // Fade/grow in when we land on the clip beat.
    const clipIntro =
      curBeat === clipStageIdx ? t : 1;
    drawClipRing(ctx, cx, cy, clipR, clipIntro, now);
  }

  const n = cur.clients.length;

  // Resolve each client's interpolated screen position.
  for (let i = 0; i < n; i++) {
    const pc = prev.clients[i];
    const cc = cur.clients[i];

    // Beat 1 "arrive": fly in from edges to xy.
    let px: number;
    let py: number;
    if (curBeat === 0) {
      const e = entries[i];
      const fly = reduceMotion ? 1 : easeOut(t);
      const sx = cc.xy[0] + e.ex * (1 - fly);
      const sy = cc.xy[1] + e.ey * (1 - fly);
      px = toX(sx);
      py = toY(sy);
    } else {
      px = lerp(toX(pc.xy[0]), toX(cc.xy[0]), t);
      py = lerp(toY(pc.xy[1]), toY(cc.xy[1]), t);
    }
    scratch[i].x = px;
    scratch[i].y = py;
  }

  // Beat 4 "aggregate": draw convergence arrows from survivors to the new global.
  const globalX = lerp(toX(prev.global[0]), toX(cur.global[0]), t);
  const globalY = lerp(toY(prev.global[1]), toY(cur.global[1]), t);
  if (stage === "aggregate") {
    const aggT = curBeat === 3 ? easeOut(t) : 1;
    ctx.save();
    ctx.lineWidth = 1.4;
    for (let i = 0; i < n; i++) {
      const cc = cur.clients[i];
      if (cc.rejected) continue;
      const p = scratch[i];
      ctx.strokeStyle = `rgba(${TEAL}, ${0.16 + 0.18 * aggT})`;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(
        lerp(p.x, globalX, aggT * 0.92),
        lerp(p.y, globalY, aggT * 0.92),
      );
      ctx.stroke();
    }
    ctx.restore();
  }

  // Draw client dots.
  for (let i = 0; i < n; i++) {
    const cc = cur.clients[i];
    const p = scratch[i];
    const poisoned = cc.poisoned;

    // "select"/"aggregate": rejected client dims & falls away.
    let dim = 1;
    let dropY = 0;
    if (cc.rejected) {
      if (stage === "select") {
        dim = lerp(1, 0.4, curBeat === 2 ? t : 1);
        dropY = lerp(0, 18, curBeat === 2 ? easeIn(t) : 1);
      } else if (stage === "aggregate") {
        dim = 0.28;
        dropY = 26;
      }
    }

    drawClient(
      ctx,
      p.x,
      p.y + dropY,
      i,
      poisoned,
      cc.rejected,
      stage,
      dim,
      now,
      curBeat,
      t,
    );

    // Annotate poison's huge norm on arrive.
    if (poisoned && stage === "arrive") {
      const a = curBeat === 0 ? t : 1;
      ctx.globalAlpha = a;
      ctx.fillStyle = `rgba(${ROSE}, 0.95)`;
      ctx.font =
        "600 11px ui-monospace, 'SF Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(`‖Δ‖ ≈ ${cc.norm.toFixed(1)}`, p.x, p.y - 16);
      ctx.fillText("poisoned", p.x, p.y - 28);
      ctx.globalAlpha = 1;
    }
  }

  // Global model marker on aggregate.
  if (stage === "aggregate") {
    const a = curBeat === 3 ? easeOut(t) : 1;
    drawGlobalMarker(ctx, globalX, globalY, a, now);
  }
}

function easeIn(t: number): number {
  return t * t;
}

function drawGrid(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  width: number,
  height: number,
) {
  ctx.save();
  ctx.strokeStyle = `rgba(${MUTED}, 0.10)`;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, cy);
  ctx.lineTo(width, cy);
  ctx.moveTo(cx, 0);
  ctx.lineTo(cx, height);
  ctx.stroke();
  ctx.restore();
}

function drawClipRing(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  r: number,
  intro: number,
  now: number,
) {
  if (r <= 0) return;
  const pulse = 0.5 + 0.5 * Math.sin(now / 700);
  ctx.save();
  // Soft fill.
  const grad = ctx.createRadialGradient(cx, cy, r * 0.2, cx, cy, r);
  grad.addColorStop(0, `rgba(${GOLD}, ${0.02 * intro})`);
  grad.addColorStop(1, `rgba(${GOLD}, ${0.07 * intro})`);
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fill();

  // Dashed ring.
  ctx.setLineDash([5, 6]);
  ctx.lineWidth = 1.4;
  ctx.strokeStyle = `rgba(${GOLD}, ${(0.45 + 0.25 * pulse) * intro})`;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  // Label.
  ctx.globalAlpha = intro;
  ctx.fillStyle = `rgba(${GOLD}, 0.9)`;
  ctx.font = "600 10px ui-monospace, 'SF Mono', monospace";
  ctx.textAlign = "left";
  ctx.fillText("DP clip", cx + r * 0.7, cy - r * 0.7);
  ctx.globalAlpha = 1;
  ctx.restore();
}

function drawClient(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  index: number,
  poisoned: boolean,
  rejected: boolean,
  stage: Frame["stage"],
  dim: number,
  now: number,
  curBeat: number,
  t: number,
) {
  const rgb = poisoned ? ROSE : TEAL;
  const radius = 7;

  ctx.save();
  ctx.globalAlpha = dim;

  // Flash red on select for the rejected node.
  const flashing = rejected && stage === "select";
  const flash = flashing ? 0.5 + 0.5 * Math.sin(now / 120) : 0;

  // Glow.
  const glowR = radius * (poisoned ? 3.4 : 2.6);
  const glow = ctx.createRadialGradient(x, y, 0, x, y, glowR);
  glow.addColorStop(0, `rgba(${rgb}, ${(poisoned ? 0.5 : 0.42) + flash * 0.3})`);
  glow.addColorStop(1, `rgba(${rgb}, 0)`);
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(x, y, glowR, 0, Math.PI * 2);
  ctx.fill();

  // Dot.
  ctx.fillStyle = `rgba(${rgb}, ${0.92})`;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();

  // Crisp ring.
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = `rgba(${rgb}, 1)`;
  ctx.stroke();

  // Node label.
  ctx.fillStyle = `rgba(${TEXT}, ${0.85 * dim + 0.1})`;
  ctx.font = "600 9px ui-monospace, 'SF Mono', monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(index), x, y);
  ctx.textBaseline = "alphabetic";

  // Rejected tag: ✕ and REJECTED on select/aggregate.
  if (rejected && (stage === "select" || stage === "aggregate")) {
    const tagA = stage === "select" && curBeat === 2 ? t : 1;
    ctx.globalAlpha = dim * tagA;
    // X mark
    ctx.strokeStyle = `rgba(${ROSE}, 1)`;
    ctx.lineWidth = 2;
    const xr = radius + 5;
    ctx.beginPath();
    ctx.moveTo(x - xr, y - xr);
    ctx.lineTo(x + xr, y + xr);
    ctx.moveTo(x + xr, y - xr);
    ctx.lineTo(x - xr, y + xr);
    ctx.stroke();
    // tag
    ctx.fillStyle = `rgba(${ROSE}, 0.95)`;
    ctx.font = "700 10px ui-monospace, 'SF Mono', monospace";
    ctx.fillText("REJECTED", x, y + radius + 16);
  }

  ctx.restore();
}

function drawGlobalMarker(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  a: number,
  now: number,
) {
  ctx.save();
  ctx.globalAlpha = a;
  const pulse = 0.5 + 0.5 * Math.sin(now / 500);

  // Bright glow.
  const glow = ctx.createRadialGradient(x, y, 0, x, y, 34);
  glow.addColorStop(0, `rgba(${GOLD}, ${0.55 + 0.2 * pulse})`);
  glow.addColorStop(1, `rgba(${GOLD}, 0)`);
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(x, y, 34, 0, Math.PI * 2);
  ctx.fill();

  // Diamond core.
  const s = 9;
  ctx.fillStyle = `rgba(${GOLD}, 1)`;
  ctx.beginPath();
  ctx.moveTo(x, y - s);
  ctx.lineTo(x + s, y);
  ctx.lineTo(x, y + s);
  ctx.lineTo(x - s, y);
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = `rgba(${TEXT}, 0.9)`;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Label.
  ctx.fillStyle = `rgba(${TEXT}, 0.95)`;
  ctx.font = "600 11px ui-monospace, 'SF Mono', monospace";
  ctx.textAlign = "center";
  ctx.fillText("global model", x, y + s + 18);
  ctx.restore();
}

function Metric({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "text" | "fed" | "silo";
}) {
  const color =
    tone === "fed"
      ? "var(--fed)"
      : tone === "silo"
        ? "var(--silo)"
        : "var(--text-primary)";
  return (
    <div className="bg-bg-surface-2 px-4 py-3">
      <p className="eyebrow" style={{ color }}>
        {label}
      </p>
      <p className="tabular mt-1 font-display text-2xl leading-none text-text-primary">
        {value}
      </p>
    </div>
  );
}

function TransportButton({
  label,
  onClick,
  children,
}: {
  label: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      onClick={onClick}
      className="rounded-full border px-3.5 py-1.5 text-[12px] font-semibold tracking-wide text-text-secondary transition-colors hover:text-text-primary"
      style={{
        borderColor: "var(--border-strong)",
        background: "var(--bg-surface-2)",
      }}
    >
      {children}
    </button>
  );
}

function Legend() {
  const items: Array<{ label: string; swatch: React.ReactNode }> = [
    {
      label: "honest client",
      swatch: <Dot color="var(--fed)" />,
    },
    {
      label: "poisoned",
      swatch: <Dot color="var(--silo)" />,
    },
    {
      label: "rejected",
      swatch: (
        <span
          aria-hidden
          className="text-[12px] font-bold leading-none"
          style={{ color: "var(--silo)" }}
        >
          ✕
        </span>
      ),
    },
    {
      label: "DP clip bound",
      swatch: (
        <span
          aria-hidden
          className="inline-block h-0 w-4 border-t border-dashed"
          style={{ borderColor: "var(--accent-gold)" }}
        />
      ),
    },
    {
      label: "global model",
      swatch: (
        <span
          aria-hidden
          className="inline-block h-2.5 w-2.5 rotate-45"
          style={{ background: "var(--accent-gold)" }}
        />
      ),
    },
  ];
  return (
    <ul className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2">
      {items.map((it) => (
        <li
          key={it.label}
          className="flex items-center gap-2 text-[11px] uppercase tracking-[0.14em] text-text-muted"
        >
          <span className="grid h-3 w-4 place-items-center">{it.swatch}</span>
          {it.label}
        </li>
      ))}
    </ul>
  );
}

function Dot({ color }: { color: string }) {
  return (
    <span
      aria-hidden
      className="inline-block h-2.5 w-2.5 rounded-full"
      style={{ background: color, boxShadow: `0 0 8px ${color}` }}
    />
  );
}

function Fallback() {
  return (
    <div
      className="grid h-[340px] w-full place-items-center rounded-[14px] border bg-bg-deep text-center sm:h-[420px]"
      style={{ borderColor: "var(--border-default)" }}
    >
      <div className="max-w-sm px-6">
        <p className="eyebrow" style={{ color: "var(--silo)" }}>
          Visualization unavailable
        </p>
        <p className="mt-2 text-[13px] leading-relaxed text-text-secondary">
          Could not load the federation round data. Differential-privacy
          clipping bounds every update and Multi-Krum rejects the single poisoned
          contribution before the honest clients aggregate into the new global
          model — no raw data ever moves.
        </p>
      </div>
    </div>
  );
}

export default memo(FederationPulse);
