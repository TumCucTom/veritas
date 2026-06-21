"use client";

import { memo, useEffect, useMemo, useRef, useState } from "react";

// ── Data schema (web/public/viz/graph.json) ────────────────────────────────
type GraphNode = {
  id: number;
  bank: number; // 0..7
  mule: boolean;
  ring: number; // 0..5 for mules, -1 for normal accounts
};

type GraphFrame = {
  round: number;
  scores: number[]; // 0..1 per node, aligned with nodes[] index
};

type GraphData = {
  bankNames: string[];
  nodes: GraphNode[];
  edges: [number, number][]; // ALL edges (node-index pairs)
  ringEdges: [number, number][]; // cross-bank mule-ring links (subset of edges)
  frames: GraphFrame[];
  meta: { banks: number; rings: number; note: string };
};

// Per-round caption — narrative arc from blind silo to federated reveal.
const CAPTIONS = [
  "Round 0 — each bank sees only its own slice.",
  "Round 1 — local signal stirs on a few accounts.",
  "Round 2 — neighbouring banks exchange suspicion.",
  "Round 4 — message-passing links the corridors.",
  "Round 6 — the cross-bank rings begin to glow.",
  "Round 9 — federation surfaces the full cross-bank rings.",
] as const;

// Tuning ──────────────────────────────────────────────────────────────────
const STEP_MS = 1300; // time per round transition
const FINAL_PAUSE_MS = 2400; // extra dwell on the last frame before looping
const HOT_THRESHOLD = 0.5; // score above which a mule account is "surfaced"

// Calm slate-teal for cool/normal state → warm amber/gold for lit mule state.
// Distinct, subtle hue per ring so the six rings stay separable when warm.
const RING_HUES = [42, 30, 18, 50, 8, 36]; // degrees, all warm (amber→rose)

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
  // `scrubbing` mirrors it for render-time UI (refs must not be read in render).
  const scrubRef = useRef<number | null>(null);
  const [scrubbing, setScrubbing] = useState(false);

  // ── Fetch own data ───────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    fetch("viz/graph.json")
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

  const totalMules = useMemo(
    () => (data ? data.nodes.reduce((n, node) => n + (node.mule ? 1 : 0), 0) : 0),
    [data],
  );
  const frameCount = data?.frames.length ?? 0;

  // ── Precompute a bank-anchored layout ONCE ────────────────────────────────
  // 8 banks sit evenly around a large circle. Each account is a small
  // deterministic-jitter cluster near its bank anchor; mules are pulled a bit
  // toward the inner edge so the cross-bank ring arcs read across the hub.
  // All positions are in normalised [0,1] space, mapped to pixels at draw time.
  const precomp = useMemo(() => {
    if (!data) return null;
    const { nodes, ringEdges, meta } = data;
    const nBanks = meta.banks;
    const n = nodes.length;

    // Bank anchor positions on a circle (normalised, centred at 0.5,0.5).
    const RING_R = 0.36; // bank-anchor radius
    const bankAngle = new Float32Array(nBanks);
    const bankAX = new Float32Array(nBanks);
    const bankAY = new Float32Array(nBanks);
    for (let b = 0; b < nBanks; b++) {
      // start at top (-90°), go clockwise
      const a = -Math.PI / 2 + (b / nBanks) * Math.PI * 2;
      bankAngle[b] = a;
      bankAX[b] = 0.5 + Math.cos(a) * RING_R;
      bankAY[b] = 0.5 + Math.sin(a) * RING_R;
    }

    // Per-node base positions (normalised).
    const nx = new Float32Array(n);
    const ny = new Float32Array(n);
    // deterministic hash jitter from node id (no RNG, stable across renders)
    const frac = (v: number) => v - Math.floor(v);
    for (let i = 0; i < n; i++) {
      const b = nodes[i].bank;
      const a = bankAngle[b];
      const h1 = frac(Math.sin(i * 12.9898 + 4.1) * 43758.5453);
      const h2 = frac(Math.sin(i * 78.233 + 1.7) * 12543.1234);
      const isMule = nodes[i].mule;
      // tangential spread along the bank arc + radial offset
      const spread = (h1 - 0.5) * 0.16; // along-arc
      // mules sit closer to centre (inner edge) so arcs bow inward cleanly
      const radial = isMule ? -0.04 - h2 * 0.05 : 0.01 + h2 * 0.085;
      const r = RING_R + radial;
      const ang = a + spread;
      nx[i] = 0.5 + Math.cos(ang) * r;
      ny[i] = 0.5 + Math.sin(ang) * r;
    }

    // Ring-edge endpoints + quadratic control point bowing toward centre.
    const re = ringEdges;
    const reA = new Int32Array(re.length);
    const reB = new Int32Array(re.length);
    const reCX = new Float32Array(re.length); // control-point x (normalised)
    const reCY = new Float32Array(re.length);
    const reHue = new Float32Array(re.length); // ring hue for tint
    for (let e = 0; e < re.length; e++) {
      const a = re[e][0];
      const b = re[e][1];
      reA[e] = a;
      reB[e] = b;
      const mx = (nx[a] + nx[b]) / 2;
      const my = (ny[a] + ny[b]) / 2;
      // pull the midpoint toward the centre (0.5,0.5) to bow the arc inward
      reCX[e] = lerp(mx, 0.5, 0.55);
      reCY[e] = lerp(my, 0.5, 0.55);
      const ring = nodes[a].ring >= 0 ? nodes[a].ring : nodes[b].ring;
      reHue[e] = RING_HUES[((ring % RING_HUES.length) + RING_HUES.length) % RING_HUES.length];
    }

    // Reusable interpolated-score buffer (allocated once).
    const interp = new Float32Array(n);
    return {
      nBanks,
      bankAX,
      bankAY,
      nx,
      ny,
      reA,
      reB,
      reCX,
      reCY,
      reHue,
      interp,
    };
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

    const { bankAX, bankAY, nx, ny, reA, reB, reCX, reCY, reHue, interp } = precomp;
    const { nodes, frames, bankNames } = data;
    const n = nodes.length;
    const nBanks = precomp.nBanks;
    const reLen = reA.length;
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

    let raf = 0;
    let visible = true;
    const pausedLocal = paused;
    let stepStart = performance.now();
    let fromFrame = reduceMotion ? lastFrame : 0; // reduced-motion holds final

    let lastUiRound = -1;
    let lastUiHot = -1;

    // Map a normalised [0,1] coord to padded device-independent pixels.
    // A square inscribed in the panel keeps the bank circle round.
    const draw = (now: number) => {
      const PAD = 30;
      const side = Math.min(width, height) - PAD * 2;
      const ox = (width - side) / 2;
      const oy = (height - side) / 2;
      const px = (v: number) => ox + v * side;
      const py = (v: number) => oy + v * side;

      // ── Resolve the interpolation target ──────────────────────────────────
      const scrub = scrubRef.current;
      let toFrame: number;
      let phase: number;

      if (scrub !== null) {
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

      // Interpolate per-node scores + tally surfaced mules in one pass.
      let hot = 0;
      for (let i = 0; i < n; i++) {
        const s = fa[i] + (fb[i] - fa[i]) * phase;
        interp[i] = s;
        if (nodes[i].mule && s > HOT_THRESHOLD) hot++;
      }

      // ── Background ────────────────────────────────────────────────────────
      ctx.clearRect(0, 0, width, height);
      const bg = ctx.createLinearGradient(0, 0, width, height);
      bg.addColorStop(0, "rgba(8,12,20,0.96)");
      bg.addColorStop(0.6, "rgba(11,16,28,0.96)");
      bg.addColorStop(1, "rgba(16,23,38,0.96)");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, width, height);

      // ── Faint bank ring guide + anchor labels ─────────────────────────────
      const cx0 = px(0.5);
      const cy0 = py(0.5);
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(120,140,170,0.07)";
      ctx.beginPath();
      ctx.arc(cx0, cy0, 0.36 * side, 0, Math.PI * 2);
      ctx.stroke();

      ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let b = 0; b < nBanks; b++) {
        const ax = px(bankAX[b]);
        const ay = py(bankAY[b]);
        // small anchor tick
        ctx.fillStyle = "rgba(150,170,200,0.30)";
        ctx.beginPath();
        ctx.arc(ax, ay, 2.2, 0, Math.PI * 2);
        ctx.fill();
        // label pushed slightly outward from centre
        const dx = bankAX[b] - 0.5;
        const dy = bankAY[b] - 0.5;
        const len = Math.hypot(dx, dy) || 1;
        const lx = px(0.5 + (dx / len) * 0.455);
        const ly = py(0.5 + (dy / len) * 0.455);
        ctx.fillStyle = "rgba(176,192,214,0.62)";
        ctx.fillText(bankNames[b] ?? `Bank ${b}`, lx, ly);
      }

      // ── Cross-bank ring arcs (the signal): curved, bowed toward centre ────
      // Brightness keyed to the hotter endpoint; round 0 → dim grey,
      // later rounds → warm gold tinted subtly by ring.
      for (let e = 0; e < reLen; e++) {
        const a = reA[e];
        const b = reB[e];
        const heat = clamp01(Math.max(interp[a], interp[b]));
        const x1 = px(nx[a]);
        const y1 = py(ny[a]);
        const x2 = px(nx[b]);
        const y2 = py(ny[b]);
        const cxp = px(reCX[e]);
        const cyp = py(reCY[e]);
        // grey when cold → warm when hot
        const sat = Math.round(lerp(6, 80, heat));
        const light = Math.round(lerp(38, 60, heat));
        const alpha = lerp(0.1, 0.62, heat);
        ctx.strokeStyle = `hsla(${reHue[e]},${sat}%,${light}%,${alpha.toFixed(3)})`;
        ctx.lineWidth = lerp(0.8, 2.0, heat);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.quadraticCurveTo(cxp, cyp, x2, y2);
        ctx.stroke();
      }

      // ── Glow pass (additive) for warm mule nodes ──────────────────────────
      ctx.globalCompositeOperation = "lighter";
      for (let i = 0; i < n; i++) {
        if (!nodes[i].mule) continue;
        const s = interp[i];
        if (s < 0.32) continue;
        const cx = px(nx[i]);
        const cy = py(ny[i]);
        const glowR = 5 + s * 20;
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, glowR);
        const a = (s - 0.3) * 0.55;
        g.addColorStop(0, `hsla(40,95%,62%,${a.toFixed(3)})`);
        g.addColorStop(1, "hsla(40,95%,62%,0)");
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(cx, cy, glowR, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalCompositeOperation = "source-over";

      // ── Node bodies ───────────────────────────────────────────────────────
      // Normal accounts: small, dim slate/teal. Mules: larger, warm, keyed
      // to score (look like normals when cold, light up gold when surfaced).
      for (let i = 0; i < n; i++) {
        const cx = px(nx[i]);
        const cy = py(ny[i]);
        if (!nodes[i].mule) {
          ctx.fillStyle = "rgba(96,126,140,0.55)";
          ctx.beginPath();
          ctx.arc(cx, cy, 2.1, 0, Math.PI * 2);
          ctx.fill();
          continue;
        }
        const s = interp[i];
        const r = 3.0 + s * 4.0;
        // cold mule reads like a normal account (cool); warms with score
        const sat = Math.round(lerp(14, 92, s));
        const light = Math.round(lerp(46, 60, s));
        ctx.fillStyle = `hsl(${lerp(200, 40, s)},${sat}%,${light}%)`;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
        // thin warm rim once surfaced
        if (s > 0.4) {
          ctx.lineWidth = 1;
          ctx.strokeStyle = `rgba(246,233,205,${(0.25 + s * 0.5).toFixed(3)})`;
          ctx.stroke();
        }
      }

      // ── Throttled UI sync ────────────────────────────────────────────────
      const uiFrame = scrub !== null ? scrub : toFrame;
      const displayRound = frames[uiFrame].round;
      if (displayRound !== lastUiRound) {
        lastUiRound = displayRound;
        setActiveRound(uiFrame);
      }
      if (hot !== lastUiHot) {
        lastUiHot = hot;
        setHotCount(hot);
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
    // Re-create loop when data/precomp ready or pause toggles.
  }, [data, precomp, paused]);

  const caption =
    CAPTIONS[Math.min(activeRound, CAPTIONS.length - 1)] ?? CAPTIONS[CAPTIONS.length - 1];
  const roundLabel =
    data && data.frames[activeRound] ? data.frames[activeRound].round : activeRound;

  // ── Fallback ────────────────────────────────────────────────────────────
  if (error) {
    return (
      <section
        aria-label="Cross-bank mule rings"
        className="rise overflow-hidden rounded-[20px] border bg-bg-surface/80 p-6 backdrop-blur-sm sm:p-8"
        style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-panel)" }}
      >
        <p className="eyebrow text-accent-gold">Federated GNN · cross-bank mule rings</p>
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
          <p className="eyebrow text-accent-gold">Federated GNN · cross-bank mule rings</p>
          <h2
            id="mule-graph-heading"
            className="mt-2 font-display text-[clamp(1.7rem,1.2rem+2vw,3rem)] leading-tight tracking-tight text-text-primary"
          >
            The ring no single bank could see.
          </h2>
          <p className="mt-3 max-w-3xl text-[13px] leading-relaxed text-text-secondary sm:text-[14px]">
            Each of {data?.meta.banks ?? 8} banks sees only its own accounts. A federated GraphSAGE
            GNN passes messages across institutions without moving a record — and the cross-bank
            mule rings light up that no bank could spot alone.
          </p>
        </div>
        <div
          className="grid grid-cols-2 gap-px overflow-hidden rounded-[14px] border text-center sm:min-w-[300px]"
          style={{ borderColor: "var(--border-default)", background: "var(--border-default)" }}
        >
          <Stat label="round" value={String(roundLabel).padStart(2, "0")} tone="gold" />
          <Stat label="mules surfaced" value={`${hotCount}/${totalMules}`} tone="fed" />
        </div>
      </header>

      <div className="p-4 sm:p-5">
        <div
          ref={wrapRef}
          className="relative h-[400px] w-full overflow-hidden rounded-[14px] border bg-bg-deep sm:h-[480px]"
          style={{ borderColor: "var(--border-default)" }}
        >
          <canvas
            ref={canvasRef}
            className="block h-full w-full"
            role="img"
            aria-label="Eight banks around a circle with cross-bank mule rings surfacing over federated rounds"
          />

          {/* Legend overlay — only colours that actually appear. */}
          <div
            className="pointer-events-none absolute left-3 top-3 flex flex-col gap-1.5 rounded-[10px] border px-3 py-2 text-[11px] backdrop-blur-sm"
            style={{
              borderColor: "var(--border-default)",
              background: "rgba(11,16,28,0.7)",
            }}
          >
            <LegendRow color="hsl(40,95%,60%)" glow label="mule account (surfaced)" />
            <LegendRow color="rgba(96,126,140,0.9)" label="normal account" />
            <LegendRow color="hsl(40,80%,58%)" line label="cross-bank mule link" />
          </div>

          {/* Caption overlay */}
          <p
            className="pointer-events-none absolute bottom-3 left-3 right-3 text-[12px] leading-snug text-text-secondary sm:text-[13px]"
            style={{ textShadow: "0 1px 8px rgba(8,12,20,0.92)" }}
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
                    setScrubbing(true);
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
            {data?.meta.rings ?? 6} cross-bank rings · {hotCount}/{totalMules} mule accounts
            surfaced
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
  glow,
  line,
}: {
  color: string;
  label: string;
  glow?: boolean;
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
            boxShadow: glow ? "0 0 6px 1px hsla(40,95%,60%,0.7)" : undefined,
          }}
        />
      )}
      {label}
    </span>
  );
}

const MuleGraph = memo(MuleGraphInner);
export default MuleGraph;
