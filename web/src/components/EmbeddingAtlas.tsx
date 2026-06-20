"use client";
import { memo, useEffect, useRef, useState } from "react";

/**
 * EmbeddingAtlas — the centerpiece "wow" visualization.
 *
 * Renders the embedding model's learned space as a canvas scatter plot and
 * auto-plays through federated training rounds, smoothly interpolating each
 * point from one frame's position to the next. Round 0 is an undifferentiated
 * blob; by the final frame the fraud (label 1) cluster pulls apart from legit.
 *
 * A "Siloed (one bank alone)" toggle morphs every point toward `siloedFinal`
 * to show the federated separation is genuinely sharper than a single bank's.
 *
 * Performance contract: one requestAnimationFrame loop held in refs (never
 * recreated on re-render), paused via IntersectionObserver when offscreen,
 * devicePixelRatio capped at 2, and zero allocation inside the tick.
 */

type Pt = [number, number];

interface Frame {
  round: number;
  fed: Pt[];
}

interface UmapData {
  n: number;
  labels: number[];
  frames: Frame[];
  siloedFinal: Pt[];
  meta: { method: string; note: string };
}

const FRAUD = "#fb7185"; // rose
const LEGIT = "#34d399"; // emerald
const TRANSITION_MS = 1200; // ~1.2s per round transition
const FINAL_HOLD_MS = 1700; // brief pause on the separated frame before looping
const SILO_MORPH_MS = 900;

// Separation-score badge (provided): higher = cleaner fraud/legit split.
const FED_SCORE = 2.66;
const SILO_SCORE = 1.69;

// One caption per round index — narrates the federation pulling fraud apart.
const CAPTIONS = [
  "entangled — banks can't tell fraud from noise alone",
  "first gradients shared — a faint seam appears",
  "the network starts to agree on what's anomalous",
  "a mule-shaped region begins to detach",
  "the scam cluster lifts away from honest traffic",
  "separation sharpens as members reinforce the signal",
  "federation pulls the scam cluster fully apart",
];

function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

function EmbeddingAtlasImpl() {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [data, setData] = useState<UmapData | null>(null);
  const [failed, setFailed] = useState(false);
  const [siloed, setSiloed] = useState(false);
  // displayRound drives the caption/indicator; mutated from the RAF loop, so it
  // lives behind a reducer-style bump rather than per-frame setState churn.
  const [displayRound, setDisplayRound] = useState(0);

  // Refs the tick reads/writes — never trigger React re-renders.
  const dataRef = useRef<UmapData | null>(null);
  const siloedRef = useRef(false);
  const visibleRef = useRef(true);
  const reducedMotionRef = useRef(false);

  // Animation state held entirely in refs so the tick allocates nothing.
  const fromIdxRef = useRef(0); // current frame index (interp source)
  const segStartRef = useRef(0); // timestamp the current segment began
  const holdingRef = useRef(false); // pausing on final frame
  const siloMixRef = useRef(0); // 0 = federated layout, 1 = siloed layout
  const siloStartRef = useRef(0);

  // Scratch buffer for resolved on-screen positions (x,y interleaved). Allocated
  // once per dataset load, never inside the tick.
  const posRef = useRef<Float32Array | null>(null);

  // Manual scrub override: when set, the loop snaps to this frame and pauses
  // auto-advance for a beat so a presenter can park on a round.
  const scrubRef = useRef<number | null>(null);

  const sizeRef = useRef({ w: 0, h: 0 });

  // --- Data fetch -----------------------------------------------------------
  useEffect(() => {
    let alive = true;
    fetch("/viz/umap.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((json: UmapData) => {
        if (!alive) return;
        if (!json?.frames?.length || !json.labels?.length) {
          throw new Error("malformed umap.json");
        }
        dataRef.current = json;
        posRef.current = new Float32Array(json.n * 2);
        setData(json);
      })
      .catch(() => {
        if (alive) setFailed(true);
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    siloedRef.current = siloed;
    siloStartRef.current = performance.now();
  }, [siloed]);

  // --- The single RAF loop --------------------------------------------------
  useEffect(() => {
    if (!data) return;
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    reducedMotionRef.current = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    const sizeCanvas = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      const w = Math.max(280, Math.floor(rect.width));
      const h = Math.max(260, Math.floor(rect.height));
      sizeRef.current.w = w;
      sizeRef.current.h = h;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    sizeCanvas();

    const ro = new ResizeObserver(sizeCanvas);
    ro.observe(canvas);

    const io = new IntersectionObserver(
      (entries) => {
        visibleRef.current = entries[0]?.isIntersecting ?? true;
      },
      { threshold: 0.05 },
    );
    io.observe(wrap);

    segStartRef.current = performance.now();
    let raf = 0;
    let lastDrawnRound = -1;

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      const d = dataRef.current;
      const pos = posRef.current;
      if (!d || !pos) return;
      if (!visibleRef.current) return; // paused offscreen — cheap idle spin

      const frames = d.frames;
      const lastIdx = frames.length - 1;

      // Resolve the siloed-morph mix (eased) from when the toggle last flipped.
      const siloTarget = siloedRef.current ? 1 : 0;
      if (siloMixRef.current !== siloTarget) {
        const sp = Math.min(1, (now - siloStartRef.current) / SILO_MORPH_MS);
        const eased = easeInOut(sp);
        siloMixRef.current = siloedRef.current ? eased : 1 - eased;
        if (sp >= 1) siloMixRef.current = siloTarget;
      }
      const silo = siloMixRef.current;

      // --- advance the federated frame interpolation ---
      let segT: number; // 0..1 progress within the current segment
      let fromIdx = fromIdxRef.current;
      let toIdx: number;

      const scrub = scrubRef.current;
      if (scrub != null) {
        // Presenter parked on a round: snap there, no interpolation.
        fromIdx = scrub;
        fromIdxRef.current = scrub;
        toIdx = scrub;
        segT = 1;
      } else if (reducedMotionRef.current) {
        // Reduced motion: hold on the final, separated frame statically.
        fromIdx = lastIdx;
        toIdx = lastIdx;
        segT = 1;
      } else {
        const dur = holdingRef.current
          ? TRANSITION_MS + FINAL_HOLD_MS
          : TRANSITION_MS;
        const elapsed = now - segStartRef.current;
        if (elapsed >= dur) {
          // segment complete — advance
          segStartRef.current = now;
          if (holdingRef.current) {
            holdingRef.current = false;
            fromIdx = 0; // loop back to the entangled blob
          } else {
            fromIdx = fromIdx + 1;
          }
          fromIdxRef.current = fromIdx;
        }
        if (fromIdx >= lastIdx) {
          // sitting on the final frame: hold, then loop
          fromIdx = lastIdx;
          fromIdxRef.current = lastIdx;
          toIdx = lastIdx;
          holdingRef.current = true;
          segT = 1;
        } else {
          toIdx = fromIdx + 1;
          const raw = Math.min(1, (now - segStartRef.current) / TRANSITION_MS);
          segT = easeInOut(raw);
        }
      }

      // Surface the current round to the caption/scrubber UI, only on change.
      const shownRound = frames[segT < 0.5 ? fromIdx : toIdx].round;
      if (shownRound !== lastDrawnRound) {
        lastDrawnRound = shownRound;
        setDisplayRound(shownRound);
      }

      // --- resolve positions: interp fed[from]→fed[to], then morph→siloed ---
      const from = frames[fromIdx].fed;
      const to = frames[toIdx].fed;
      const siloFinal = d.siloedFinal;
      const n = d.n;
      for (let i = 0; i < n; i++) {
        const fx = from[i][0];
        const fy = from[i][1];
        let x = fx + (to[i][0] - fx) * segT;
        let y = fy + (to[i][1] - fy) * segT;
        if (silo > 0) {
          x = x + (siloFinal[i][0] - x) * silo;
          y = y + (siloFinal[i][1] - y) * silo;
        }
        pos[i * 2] = x;
        pos[i * 2 + 1] = y;
      }

      drawScene(ctx, d, pos, sizeRef.current.w, sizeRef.current.h);
    };

    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
    };
    // The loop is intentionally created once per dataset; it reads live state
    // from refs, so it must NOT re-run on toggle/scrub changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  // --- Scrub handling: park on a round briefly, then resume autoplay ---------
  const scrubTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const handleScrub = (idx: number) => {
    if (!dataRef.current) return;
    scrubRef.current = idx;
    setDisplayRound(dataRef.current.frames[idx].round);
    if (scrubTimer.current) clearTimeout(scrubTimer.current);
    scrubTimer.current = setTimeout(() => {
      // Resume the autoplay loop from the parked frame.
      fromIdxRef.current = idx;
      segStartRef.current = performance.now();
      holdingRef.current = idx >= (dataRef.current!.frames.length - 1);
      scrubRef.current = null;
    }, 2600);
  };
  useEffect(
    () => () => {
      if (scrubTimer.current) clearTimeout(scrubTimer.current);
    },
    [],
  );

  // --- Fallback -------------------------------------------------------------
  if (failed) {
    return (
      <section
        aria-label="Learned embedding space"
        className="rise overflow-hidden rounded-[20px] border bg-bg-surface/80 p-6 backdrop-blur-sm sm:p-8"
        style={{
          borderColor: "var(--border-default)",
          boxShadow: "var(--shadow-panel)",
        }}
      >
        <p className="eyebrow text-accent-gold">
          Learned embedding space · live from the model
        </p>
        <h2 className="mt-2 font-display text-[clamp(1.5rem,1.1rem+1.6vw,2.4rem)] leading-tight tracking-tight text-text-primary">
          Federation teaches the network what a mule looks like.
        </h2>
        <p className="mt-3 max-w-2xl text-[13px] leading-relaxed text-text-secondary">
          The embedding projection is unavailable right now. The federated model
          still separates fraud from legitimate traffic far more cleanly than any
          single bank can alone.
        </p>
      </section>
    );
  }

  const captionIdx = data
    ? Math.max(
        0,
        data.frames.findIndex((f) => f.round === displayRound),
      )
    : 0;
  const caption = siloed
    ? "one bank alone — the fraud cluster stays blurred into the crowd"
    : CAPTIONS[Math.min(captionIdx, CAPTIONS.length - 1)];

  return (
    <section
      ref={wrapRef}
      aria-labelledby="embedding-atlas-heading"
      className="rise overflow-hidden rounded-[20px] border bg-bg-surface/80 backdrop-blur-sm"
      style={{
        borderColor: "var(--border-default)",
        boxShadow: "var(--shadow-panel)",
      }}
    >
      <header
        className="grid gap-4 border-b p-5 sm:p-6 lg:grid-cols-[1fr_auto] lg:items-end"
        style={{ borderColor: "var(--border-default)" }}
      >
        <div>
          <p className="eyebrow text-accent-gold">
            Learned embedding space · live from the model
          </p>
          <h2
            id="embedding-atlas-heading"
            className="mt-2 font-display text-[clamp(1.7rem,1.2rem+2vw,3rem)] leading-tight tracking-tight text-text-primary"
          >
            Federation teaches the network what a mule looks like.
          </h2>
          <p className="mt-3 max-w-3xl text-[13px] leading-relaxed text-text-secondary sm:text-[14px]">
            Each dot is a customer projected into the model&apos;s learned
            embedding space. As federated rounds progress, the network agrees on
            what anomalous looks like and the{" "}
            <span style={{ color: FRAUD }}>fraud</span> cluster pulls away from{" "}
            <span style={{ color: LEGIT }}>legitimate</span> traffic — a
            separation no single bank reaches alone.
          </p>
        </div>

        {/* Federated vs Siloed separation-score badge. */}
        <div
          className="grid grid-cols-2 gap-px overflow-hidden rounded-[14px] border text-center sm:min-w-[280px]"
          style={{
            borderColor: "var(--border-default)",
            background: "var(--border-default)",
          }}
        >
          <div className="bg-bg-surface-2 px-4 py-3">
            <p className="eyebrow" style={{ color: "var(--fed)" }}>
              Federated
            </p>
            <p className="tabular mt-1 font-display text-2xl leading-none text-text-primary">
              {FED_SCORE.toFixed(2)}
            </p>
            <p className="mt-1 text-[10px] text-text-muted">separation</p>
          </div>
          <div className="bg-bg-surface-2 px-4 py-3">
            <p className="eyebrow" style={{ color: "var(--silo)" }}>
              Siloed
            </p>
            <p className="tabular mt-1 font-display text-2xl leading-none text-text-primary">
              {SILO_SCORE.toFixed(2)}
            </p>
            <p className="mt-1 text-[10px] text-text-muted">separation</p>
          </div>
        </div>
      </header>

      <div className="p-4 sm:p-5">
        {/* Top bar: legend + round indicator + siloed toggle. */}
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-4">
            <Legend color={LEGIT} label="legit" />
            <Legend color={FRAUD} label="fraud" />
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <span className="eyebrow">round</span>{" "}
              <span className="tabular font-display text-xl leading-none text-text-primary">
                {String(displayRound).padStart(2, "0")}
              </span>
            </div>
            <button
              type="button"
              onClick={() => setSiloed((s) => !s)}
              aria-pressed={siloed}
              className="rounded-[10px] border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.18em] transition-colors"
              style={{
                borderColor: siloed ? "var(--silo)" : "var(--border-strong)",
                color: siloed ? "var(--silo)" : "var(--text-secondary)",
                background: siloed ? "rgba(240,74,82,0.08)" : "transparent",
              }}
            >
              {siloed ? "Siloed view · on" : "Siloed (one bank alone)"}
            </button>
          </div>
        </div>

        <canvas
          ref={canvasRef}
          className="block h-[360px] w-full rounded-[14px] border bg-bg-deep sm:h-[440px]"
          style={{ borderColor: "var(--border-default)" }}
          role="img"
          aria-label="Scatter plot of customers in the learned embedding space, fraud separating from legitimate traffic across federated rounds"
        />

        {/* Caption + round scrubber. */}
        <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <p
            className="text-[13px] leading-relaxed text-text-secondary"
            aria-live="polite"
          >
            <span
              className="font-display text-text-primary"
              style={{ color: siloed ? "var(--silo)" : undefined }}
            >
              {siloed ? "Siloed" : `Round ${String(displayRound).padStart(2, "0")}`}
            </span>{" "}
            — {caption}
          </p>

          <div
            className="flex items-center gap-1.5"
            role="group"
            aria-label="Step through federated rounds"
          >
            {(data?.frames ?? []).map((f, i) => {
              const active = f.round === displayRound && !siloed;
              return (
                <button
                  key={f.round}
                  type="button"
                  onClick={() => handleScrub(i)}
                  aria-label={`Go to round ${f.round}`}
                  aria-pressed={active}
                  className="group flex flex-col items-center gap-1"
                >
                  <span
                    className="h-2 w-2 rounded-full transition-all"
                    style={{
                      background: active
                        ? "var(--accent-gold)"
                        : "var(--border-strong)",
                      transform: active ? "scale(1.4)" : "scale(1)",
                    }}
                  />
                  <span
                    className="tabular text-[10px] leading-none transition-colors"
                    style={{
                      color: active ? "var(--accent-gold)" : "var(--text-muted)",
                    }}
                  >
                    {f.round}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span
        className="h-2.5 w-2.5 rounded-full"
        style={{ background: color, boxShadow: `0 0 8px ${color}66` }}
        aria-hidden
      />
      <span className="eyebrow" style={{ color }}>
        {label}
      </span>
    </span>
  );
}

/**
 * Draws one frame. Allocation-free: no arrays/objects created here, gradients
 * for the soft fraud glow are the only per-call canvas objects (cheap, and only
 * paid by the fraud minority). Coordinates arrive in [-1,1].
 */
type BBox = { minX: number; maxX: number; minY: number; maxY: number };
const bboxCache = new WeakMap<UmapData, BBox>();
function getBBox(d: UmapData): BBox {
  const cached = bboxCache.get(d);
  if (cached) return cached;
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  const scan = (pts: number[][]) => {
    for (let i = 0; i < pts.length; i++) {
      const x = pts[i][0];
      const y = pts[i][1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  };
  for (const f of d.frames) scan(f.fed);
  if (d.siloedFinal) scan(d.siloedFinal);
  const bb: BBox = { minX, maxX, minY, maxY };
  bboxCache.set(d, bb);
  return bb;
}

function drawScene(
  ctx: CanvasRenderingContext2D,
  d: UmapData,
  pos: Float32Array,
  w: number,
  h: number,
) {
  ctx.clearRect(0, 0, w, h);

  // Subtle radial vignette matching the site's deep canvas.
  const bg = ctx.createRadialGradient(
    w * 0.5,
    h * 0.45,
    Math.min(w, h) * 0.1,
    w * 0.5,
    h * 0.5,
    Math.max(w, h) * 0.75,
  );
  bg.addColorStop(0, "rgba(19,27,44,0.0)");
  bg.addColorStop(1, "rgba(10,14,22,0.55)");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  // Map the data's ACTUAL bounding box (across all frames) → padded canvas,
  // centered with a uniform scale so the point cloud fills the panel instead of
  // collapsing into a sliver when the UMAP range isn't centered on the origin.
  const pad = 26;
  const bb = getBBox(d);
  const cx = (bb.minX + bb.maxX) / 2;
  const cy = (bb.minY + bb.maxY) / 2;
  const dw = Math.max(1e-6, bb.maxX - bb.minX);
  const dh = Math.max(1e-6, bb.maxY - bb.minY);
  const scale = Math.min((w - pad * 2) / dw, (h - pad * 2) / dh);
  const sx = scale;
  const sy = scale;
  const ox = w / 2 - cx * scale;
  const oy = h / 2 - cy * scale;

  const n = d.n;
  const labels = d.labels;
  const r = w > 620 ? 2.4 : 2.0;

  // Pass 1 — legit (emerald). Single fillStyle, soft constant alpha.
  ctx.fillStyle = LEGIT;
  ctx.globalAlpha = 0.7;
  for (let i = 0; i < n; i++) {
    if (labels[i] === 1) continue;
    const px = ox + pos[i * 2] * sx;
    const py = oy + pos[i * 2 + 1] * sy;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fill();
  }

  // Pass 2 — fraud (rose) with a subtle additive glow.
  ctx.globalCompositeOperation = "lighter";
  for (let i = 0; i < n; i++) {
    if (labels[i] !== 1) continue;
    const px = ox + pos[i * 2] * sx;
    const py = oy + pos[i * 2 + 1] * sy;
    const glow = ctx.createRadialGradient(px, py, 0, px, py, r * 3.4);
    glow.addColorStop(0, "rgba(251,113,133,0.55)");
    glow.addColorStop(1, "rgba(251,113,133,0)");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(px, py, r * 3.4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
  ctx.fillStyle = FRAUD;
  for (let i = 0; i < n; i++) {
    if (labels[i] !== 1) continue;
    const px = ox + pos[i * 2] * sx;
    const py = oy + pos[i * 2 + 1] * sy;
    ctx.beginPath();
    ctx.arc(px, py, r * 1.05, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

const EmbeddingAtlas = memo(EmbeddingAtlasImpl);
export default EmbeddingAtlas;
