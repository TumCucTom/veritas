"use client";
import { memo, useEffect, useRef, useState } from "react";

/**
 * EmbeddingAtlas — the centerpiece "wow" visualization, now in interactive 3D.
 *
 * Renders the embedding model's learned space as a hand-rolled 3D canvas point
 * cloud (no three.js / regl — the repo stays dependency-free). Each point is
 * rotated by a yaw+pitch camera, perspective-projected, depth-sorted back to
 * front, and drawn with depth-scaled radius/alpha + a soft fraud glow so the
 * cloud reads as volumetric. Round 0 is an undifferentiated blob; by the final
 * frame the fraud (label 1) cluster pulls apart from legit.
 *
 * INTERACTION
 *  - Orbit: pointer drag rotates yaw/pitch; slow idle auto-rotation resumes a
 *    couple seconds after release. Touch-friendly via pointer capture.
 *  - Round selection: a row of round buttons + a Play/Pause control. Picking a
 *    round eases to it and HOLDS so the user can orbit it; Play auto-advances,
 *    smoothly interpolating point positions between consecutive frames.
 *  - Siloed A/B toggle morphs every point toward `siloedFinal`.
 *
 * Performance contract: one requestAnimationFrame loop held in refs (never
 * recreated on re-render), paused via IntersectionObserver when offscreen,
 * devicePixelRatio capped at 2, and zero allocation inside the tick.
 */

type Pt3 = [number, number, number];

interface Frame {
  round: number;
  fed: Pt3[];
}

interface UmapData {
  n: number;
  labels: number[];
  frames: Frame[];
  siloedFinal: Pt3[];
  meta: { method: string; dims: number; note: string };
}

const FRAUD = "#fb7185"; // rose
const LEGIT = "#3ddc97"; // emerald
const TRANSITION_MS = 1200; // ~1.2s per round transition
const FINAL_HOLD_MS = 1700; // pause on the separated frame before looping
const SILO_MORPH_MS = 900;
const ROUND_EASE_MS = 1100; // ease when a user picks a round
const IDLE_RESUME_MS = 2200; // resume auto-rotate this long after letting go
const AUTO_YAW_SPEED = 0.18; // rad/s idle spin
const FOCAL = 3.2; // perspective focal / camera distance
const CAM_Z = 3.2;

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
function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function EmbeddingAtlasImpl() {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [data, setData] = useState<UmapData | null>(null);
  const [failed, setFailed] = useState(false);
  const [siloed, setSiloed] = useState(false);
  const [playing, setPlaying] = useState(true);
  // displayRound drives the caption/indicator; mutated from the RAF loop, so it
  // lives behind a reducer-style bump rather than per-frame setState churn.
  const [displayRound, setDisplayRound] = useState(0);

  // Refs the tick reads/writes — never trigger React re-renders.
  const dataRef = useRef<UmapData | null>(null);
  const siloedRef = useRef(false);
  const playingRef = useRef(true);
  const visibleRef = useRef(true);
  const reducedMotionRef = useRef(false);

  // Frame interpolation state (refs so the tick allocates nothing).
  const fromIdxRef = useRef(0); // current frame index (interp source)
  const segStartRef = useRef(0); // timestamp the current segment began
  const holdingRef = useRef(false); // pausing on final frame during autoplay
  const siloMixRef = useRef(0); // 0 = federated layout, 1 = siloed layout
  const siloStartRef = useRef(0);

  // Round-selection (manual hold) easing. When selectedRef != null we ease from
  // selEaseFromRef → selectedRef and hold there until the user plays again.
  const selectedRef = useRef<number | null>(null);
  const selEaseFromRef = useRef(0);
  const selStartRef = useRef(0);

  // Camera orbit state.
  const yawRef = useRef(0.6);
  const pitchRef = useRef(0.35);
  const draggingRef = useRef(false);
  const lastPtrRef = useRef({ x: 0, y: 0 });
  const velRef = useRef({ yaw: 0, pitch: 0 }); // inertia
  const lastInteractRef = useRef(0); // perf.now() of last user interaction
  const lastTickRef = useRef(0); // for dt-based auto-rotate / inertia

  // Preallocated buffers — allocated once per dataset, never inside the tick.
  const pos3Ref = useRef<Float32Array | null>(null); // interpolated x,y,z
  const rotRef = useRef<Float32Array | null>(null); // rotated camera-space x,y,z
  const projRef = useRef<Float32Array | null>(null); // screen x,y + depth scale
  const orderRef = useRef<Int32Array | null>(null); // depth-sorted indices

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
        pos3Ref.current = new Float32Array(json.n * 3);
        rotRef.current = new Float32Array(json.n * 3);
        projRef.current = new Float32Array(json.n * 3);
        const order = new Int32Array(json.n);
        for (let i = 0; i < json.n; i++) order[i] = i;
        orderRef.current = order;
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

  useEffect(() => {
    playingRef.current = playing;
    if (playing) {
      // Resuming autoplay: clear any manual hold and continue from where we are.
      selectedRef.current = null;
      segStartRef.current = performance.now();
      holdingRef.current =
        fromIdxRef.current >= (dataRef.current?.frames.length ?? 1) - 1;
    }
  }, [playing]);

  // Select a specific round: ease to it and hold (pauses autoplay).
  const selectRound = (idx: number) => {
    if (!dataRef.current) return;
    selEaseFromRef.current = fromIdxRef.current;
    selectedRef.current = idx;
    selStartRef.current = performance.now();
    lastInteractRef.current = performance.now();
    setPlaying(false);
    playingRef.current = false;
    setDisplayRound(dataRef.current.frames[idx].round);
  };

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
    if (reducedMotionRef.current) {
      // Hold the final, separated round; no autoplay / auto-rotate.
      playingRef.current = false;
      setPlaying(false);
      fromIdxRef.current = data.frames.length - 1;
    }

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
    lastTickRef.current = performance.now();
    let raf = 0;
    let lastDrawnRound = -1;

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      const d = dataRef.current;
      const pos = pos3Ref.current;
      if (!d || !pos) return;
      if (!visibleRef.current) {
        lastTickRef.current = now;
        return; // paused offscreen
      }
      const dt = Math.min(0.05, (now - lastTickRef.current) / 1000);
      lastTickRef.current = now;

      const frames = d.frames;
      const lastIdx = frames.length - 1;

      // --- camera: drag, inertia, idle auto-rotate ---
      if (draggingRef.current) {
        // velocity applied directly in pointer handler; nothing here.
      } else {
        // inertia decay
        const v = velRef.current;
        if (Math.abs(v.yaw) > 1e-4 || Math.abs(v.pitch) > 1e-4) {
          yawRef.current += v.yaw;
          pitchRef.current = clamp(pitchRef.current + v.pitch, -1.396, 1.396);
          v.yaw *= 0.9;
          v.pitch *= 0.9;
        }
        const idleFor = now - lastInteractRef.current;
        if (
          !reducedMotionRef.current &&
          idleFor > IDLE_RESUME_MS &&
          Math.abs(v.yaw) <= 1e-4
        ) {
          yawRef.current += AUTO_YAW_SPEED * dt;
        }
      }

      // --- resolve the siloed-morph mix (eased) ---
      const siloTarget = siloedRef.current ? 1 : 0;
      if (siloMixRef.current !== siloTarget) {
        const sp = Math.min(1, (now - siloStartRef.current) / SILO_MORPH_MS);
        const eased = easeInOut(sp);
        siloMixRef.current = siloedRef.current ? eased : 1 - eased;
        if (sp >= 1) siloMixRef.current = siloTarget;
      }
      const silo = siloMixRef.current;

      // --- advance the federated frame interpolation ---
      let fromIdx = fromIdxRef.current;
      let toIdx: number;
      let segT: number; // 0..1 progress within the current segment

      const selected = selectedRef.current;
      if (selected != null) {
        // Manual hold: ease from the frame we were on toward the picked round.
        const sp = Math.min(1, (now - selStartRef.current) / ROUND_EASE_MS);
        const e = easeInOut(sp);
        fromIdx = selEaseFromRef.current;
        toIdx = selected;
        segT = e;
        if (sp >= 1) {
          fromIdxRef.current = selected;
          fromIdx = selected;
          toIdx = selected;
          segT = 1;
        }
      } else if (!playingRef.current) {
        // Paused without an active ease: hold on the current frame.
        toIdx = fromIdx;
        segT = 1;
      } else {
        const dur = holdingRef.current
          ? TRANSITION_MS + FINAL_HOLD_MS
          : TRANSITION_MS;
        if (now - segStartRef.current >= dur) {
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
          fromIdx = lastIdx;
          fromIdxRef.current = lastIdx;
          toIdx = lastIdx;
          holdingRef.current = true;
          segT = 1;
        } else {
          toIdx = fromIdx + 1;
          segT = easeInOut(
            Math.min(1, (now - segStartRef.current) / TRANSITION_MS),
          );
        }
      }

      // Surface the current round to the caption/UI, only on change.
      const shownRound = frames[segT < 0.5 ? fromIdx : toIdx].round;
      if (shownRound !== lastDrawnRound) {
        lastDrawnRound = shownRound;
        setDisplayRound(shownRound);
      }

      // --- resolve 3D positions: interp fed[from]→fed[to], then morph→siloed ---
      const from = frames[fromIdx].fed;
      const to = frames[toIdx].fed;
      const siloFinal = d.siloedFinal;
      const n = d.n;
      for (let i = 0; i < n; i++) {
        const f = from[i];
        const t = to[i];
        let x = f[0] + (t[0] - f[0]) * segT;
        let y = f[1] + (t[1] - f[1]) * segT;
        let z = f[2] + (t[2] - f[2]) * segT;
        if (silo > 0) {
          const s = siloFinal[i];
          x += (s[0] - x) * silo;
          y += (s[1] - y) * silo;
          z += (s[2] - z) * silo;
        }
        pos[i * 3] = x;
        pos[i * 3 + 1] = y;
        pos[i * 3 + 2] = z;
      }

      drawScene(
        ctx,
        d,
        pos,
        rotRef.current!,
        projRef.current!,
        orderRef.current!,
        yawRef.current,
        pitchRef.current,
        sizeRef.current.w,
        sizeRef.current.h,
      );
    };

    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
    };
    // Loop is created once per dataset; it reads live state from refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  // --- Pointer orbit handlers ----------------------------------------------
  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (reducedMotionRef.current) return;
    draggingRef.current = true;
    lastPtrRef.current = { x: e.clientX, y: e.clientY };
    velRef.current = { yaw: 0, pitch: 0 };
    lastInteractRef.current = performance.now();
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!draggingRef.current) return;
    const dx = e.clientX - lastPtrRef.current.x;
    const dy = e.clientY - lastPtrRef.current.y;
    lastPtrRef.current = { x: e.clientX, y: e.clientY };
    const yawD = dx * 0.008;
    const pitchD = dy * 0.008;
    yawRef.current += yawD;
    pitchRef.current = clamp(pitchRef.current + pitchD, -1.396, 1.396);
    velRef.current = { yaw: yawD, pitch: pitchD };
    lastInteractRef.current = performance.now();
  };
  const endDrag = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    lastInteractRef.current = performance.now();
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      /* capture may already be gone */
    }
  };

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
            embedding space — now a fully 3D cloud you can orbit. As federated
            rounds progress, the network agrees on what anomalous looks like and
            the <span style={{ color: FRAUD }}>fraud</span> cluster pulls away
            from <span style={{ color: LEGIT }}>legitimate</span> traffic — a
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
            <span className="hidden text-[11px] text-text-muted sm:inline">
              drag to rotate · pick a round
            </span>
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
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          className="block h-[440px] w-full touch-none cursor-grab rounded-[14px] border bg-bg-deep active:cursor-grabbing sm:h-[560px]"
          style={{ borderColor: "var(--border-default)" }}
          role="img"
          aria-label="Interactive 3D scatter plot of customers in the learned embedding space; drag to orbit, fraud separating from legitimate traffic across federated rounds"
        />

        {/* Caption. */}
        <div className="mt-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <p
            className="text-[13px] leading-relaxed text-text-secondary"
            aria-live="polite"
          >
            <span
              className="font-display text-text-primary"
              style={{ color: siloed ? "var(--silo)" : undefined }}
            >
              {siloed
                ? "Siloed"
                : `Round ${String(displayRound).padStart(2, "0")}`}
            </span>{" "}
            — {caption}
          </p>
          <span className="text-[11px] text-text-muted sm:hidden">
            drag to rotate · pick a round
          </span>
        </div>

        {/* Round-selection control: Play/Pause + a button per frame. */}
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setPlaying((p) => !p)}
            aria-pressed={playing}
            aria-label={playing ? "Pause auto-advance" : "Play auto-advance"}
            className="inline-flex items-center gap-1.5 rounded-[10px] border px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] transition-colors"
            style={{
              borderColor: playing
                ? "var(--accent-gold)"
                : "var(--border-strong)",
              color: playing ? "var(--accent-gold)" : "var(--text-secondary)",
              background: playing ? "rgba(212,175,90,0.08)" : "transparent",
            }}
          >
            <span aria-hidden>{playing ? "❚❚" : "▶"}</span>
            {playing ? "Pause" : "Play"}
          </button>

          <div
            className="flex flex-wrap items-center gap-1.5"
            role="group"
            aria-label="Select a federated round"
          >
            {(data?.frames ?? []).map((f, i) => {
              const active = f.round === displayRound && !siloed;
              return (
                <button
                  key={f.round}
                  type="button"
                  onClick={() => selectRound(i)}
                  aria-label={`Go to round ${f.round}`}
                  aria-pressed={active}
                  className="tabular min-w-[34px] rounded-[9px] border px-2 py-1 text-[12px] font-semibold leading-none transition-all"
                  style={{
                    borderColor: active
                      ? "var(--accent-gold)"
                      : "var(--border-strong)",
                    color: active
                      ? "var(--accent-gold)"
                      : "var(--text-muted)",
                    background: active
                      ? "rgba(212,175,90,0.10)"
                      : "transparent",
                    transform: active ? "scale(1.06)" : "scale(1)",
                  }}
                >
                  {f.round}
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
 * Bounding box across all frames + siloedFinal, in 3D, cached per dataset so the
 * cloud is centered + uniformly scaled (largest of the 3 dims wins) to fill the
 * panel regardless of camera angle.
 */
type BBox3 = {
  cx: number;
  cy: number;
  cz: number;
  half: number; // half-extent of the largest axis
};
const bboxCache = new WeakMap<UmapData, BBox3>();
function getBBox(d: UmapData): BBox3 {
  const cached = bboxCache.get(d);
  if (cached) return cached;
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity,
    minZ = Infinity,
    maxZ = -Infinity;
  const scan = (pts: Pt3[]) => {
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      if (p[0] < minX) minX = p[0];
      if (p[0] > maxX) maxX = p[0];
      if (p[1] < minY) minY = p[1];
      if (p[1] > maxY) maxY = p[1];
      if (p[2] < minZ) minZ = p[2];
      if (p[2] > maxZ) maxZ = p[2];
    }
  };
  for (const f of d.frames) scan(f.fed);
  if (d.siloedFinal) scan(d.siloedFinal);
  const half =
    Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-6) / 2;
  const bb: BBox3 = {
    cx: (minX + maxX) / 2,
    cy: (minY + maxY) / 2,
    cz: (minZ + maxZ) / 2,
    half,
  };
  bboxCache.set(d, bb);
  return bb;
}

/**
 * Draws one frame in 3D. Allocation-free except the per-fraud radial gradient
 * (the fraud minority only). Steps: center/normalize → yaw+pitch rotate →
 * perspective project → depth-sort back-to-front → draw with depth-scaled
 * radius/alpha + soft fraud glow + faint depth fog.
 */
function drawScene(
  ctx: CanvasRenderingContext2D,
  d: UmapData,
  pos: Float32Array,
  rot: Float32Array,
  proj: Float32Array,
  order: Int32Array,
  yaw: number,
  pitch: number,
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

  const bb = getBBox(d);
  const inv = 1 / bb.half; // normalize so the largest axis spans ~[-1,1]
  const cy = Math.cos(yaw),
    sy = Math.sin(yaw);
  const cp = Math.cos(pitch),
    sp = Math.sin(pitch);

  const pad = 28;
  // Fill the panel: the cloud is normalized to [-1,1] on its largest axis, and
  // perspective shrinks it ~0.5 at the center, so scale up generously while
  // leaving headroom for the long axis swinging through depth on orbit.
  const fit = Math.min(w - pad * 2, h - pad * 2) * 0.82;
  const ox = w / 2;
  const oy = h / 2;
  const n = d.n;

  for (let i = 0; i < n; i++) {
    // center + normalize
    const x0 = (pos[i * 3] - bb.cx) * inv;
    const y0 = (pos[i * 3 + 1] - bb.cy) * inv;
    const z0 = (pos[i * 3 + 2] - bb.cz) * inv;
    // yaw around Y
    const xz = x0 * cy + z0 * sy;
    const zz = -x0 * sy + z0 * cy;
    // pitch around X
    const yz = y0 * cp - zz * sp;
    const zc = y0 * sp + zz * cp;
    rot[i * 3] = xz;
    rot[i * 3 + 1] = yz;
    rot[i * 3 + 2] = zc;
    // perspective project (camera looks down -z; zCam = camera distance - zc)
    const persp = FOCAL / (FOCAL + CAM_Z - zc * 1.4);
    proj[i * 3] = ox + xz * fit * persp;
    proj[i * 3 + 1] = oy - yz * fit * persp;
    proj[i * 3 + 2] = persp; // store perspective scale for radius/alpha
  }

  // Depth sort back-to-front (smaller zc = farther) — reuse the index array.
  // Insertion sort is fine and stable for n≈600 of mostly-sorted frames.
  for (let a = 1; a < n; a++) {
    const idx = order[a];
    const key = rot[idx * 3 + 2];
    let b = a - 1;
    while (b >= 0 && rot[order[b] * 3 + 2] > key) {
      order[b + 1] = order[b];
      b--;
    }
    order[b + 1] = idx;
  }

  const labels = d.labels;
  const baseR = w > 620 ? 2.5 : 2.1;

  // Single pass, back-to-front, so nearer points draw over farther ones.
  for (let k = 0; k < n; k++) {
    const i = order[k];
    const persp = proj[i * 3 + 2];
    const px = proj[i * 3];
    const py = proj[i * 3 + 1];
    // depth term: 0 (far) → 1 (near), used for fog + size.
    const depth = clamp((persp - 0.55) / 0.6, 0, 1);
    const r = baseR * (0.62 + 0.55 * depth);

    if (labels[i] === 1) {
      // fraud — soft additive glow then a solid rose core.
      ctx.globalCompositeOperation = "lighter";
      const g = r * 3.4;
      const glow = ctx.createRadialGradient(px, py, 0, px, py, g);
      const ga = 0.32 + 0.3 * depth;
      glow.addColorStop(0, `rgba(251,113,133,${ga.toFixed(3)})`);
      glow.addColorStop(1, "rgba(251,113,133,0)");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(px, py, g, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 0.7 + 0.3 * depth;
      ctx.fillStyle = FRAUD;
      ctx.beginPath();
      ctx.arc(px, py, r * 1.05, 0, Math.PI * 2);
      ctx.fill();
    } else {
      // legit — emerald with depth fog (far points dim toward the bg).
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 0.32 + 0.46 * depth;
      ctx.fillStyle = LEGIT;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
}

const EmbeddingAtlas = memo(EmbeddingAtlasImpl);
export default EmbeddingAtlas;
