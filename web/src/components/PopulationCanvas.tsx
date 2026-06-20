"use client";
import { memo, useEffect, useRef } from "react";

// Number of dots painted per regime panel. Exported so callers can derive an
// honest "customers per dot" caption from the live customer total instead of
// inventing a ratio.
export const POPULATION_DOTS = 6000;
const N = POPULATION_DOTS;
const COLS = 100;
const ROWS = Math.ceil(N / COLS);
const CANVAS_H = 220;

// State 0 = unexposed (neutral), 1 = victimised (red), 2 = protected (green).
// Stored as packed 0xAABBGGRR words so the dot grid can be blitted with a single
// putImageData instead of thousands of fillRect calls.
const RGBA = {
  neutral: pack(0x2a, 0x33, 0x46),
  atRisk: pack(0xf0, 0x4a, 0x52),
  protected: pack(0x34, 0xd3, 0x99),
} as const;

function pack(r: number, g: number, b: number): number {
  // Little-endian: 0xAABBGGRR. Alpha fixed at 0xff (opaque).
  return (0xff << 24) | (b << 16) | (g << 8) | r;
}

// Per-frame transition rates. The red<->green equilibrium is tuned so the
// steady-state share of *protected* dots equals the regime's detection rate:
// red->green ∝ detection, green->red ∝ (1 - detection). The stationary green
// fraction is therefore exactly `detection`, so the siloed panel keeps a
// visible red band (low detection) while the federated panel clears to green.
// That persistent gap is the whole story — without the relapse term both
// regimes would simply accumulate green over time and the contrast would wash out.
const EXPOSE = 0.025; // neutral -> at-risk as the campaign reaches customers
const CHURN = 0.06; // base rate of the protection dynamics

type Tone = "fed" | "silo";

interface PopulationCanvasProps {
  detection: number;
  tone: Tone;
}

// Geometry for one dot cell, in backing-store (device) pixels. Precomputed once
// per (size, dpr) so the rAF tick allocates nothing and computes no layout.
interface Grid {
  dpr: number;
  bw: number; // backing-store width in px
  bh: number; // backing-store height in px
  cw: number; // cell width in px
  ch: number; // cell height in px
  dw: number; // drawn dot width (cw minus the 1px gutter)
  dh: number; // drawn dot height
  // Per-dot top-left pixel offset into the buffer, precomputed.
  ox: Int32Array;
  oy: Int32Array;
}

function buildGrid(cssWidth: number, dpr: number): Grid {
  const bw = Math.max(1, Math.floor(cssWidth * dpr));
  const bh = Math.max(1, Math.floor(CANVAS_H * dpr));
  const cw = bw / COLS;
  const ch = bh / ROWS;
  // Keep the historical 1px gutter between dots, scaled by dpr, but never let a
  // dot collapse to zero pixels.
  const gap = Math.max(1, Math.round(dpr));
  const dw = Math.max(1, Math.round(cw) - gap);
  const dh = Math.max(1, Math.round(ch) - gap);
  const ox = new Int32Array(N);
  const oy = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    ox[i] = Math.floor((i % COLS) * cw);
    oy[i] = Math.floor(Math.floor(i / COLS) * ch);
  }
  return { dpr, bw, bh, cw, ch, dw, dh, ox, oy };
}

// Blit the whole dot grid into the pixel buffer in one pass, then upload it with
// a single putImageData. This replaces ~6000 fillStyle+fillRect calls per frame
// (12000 across both panels) — the dominant cost in the old loop — with one
// typed-array fill loop and one GPU upload.
function blit(
  ctx: CanvasRenderingContext2D,
  image: ImageData,
  buf: Uint32Array,
  st: Uint8Array,
  grid: Grid,
): void {
  buf.fill(0); // transparent background; the canvas sits on a dark panel
  const { bw, dw, dh, ox, oy } = grid;
  for (let i = 0; i < N; i++) {
    const color = st[i] === 1 ? RGBA.atRisk : st[i] === 2 ? RGBA.protected : RGBA.neutral;
    const x0 = ox[i];
    const y0 = oy[i];
    for (let y = 0; y < dh; y++) {
      let p = (y0 + y) * bw + x0;
      for (let x = 0; x < dw; x++) buf[p++] = color;
    }
  }
  ctx.putImageData(image, 0, 0);
}

function PopulationCanvas({ detection, tone }: PopulationCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const det = useRef(detection);

  // Keep the live detection value in a ref so the rAF loop reads the latest
  // figure without restarting the simulation. Synced in an effect (never during
  // render), so a new detection value never tears down or recreates the loop.
  useEffect(() => {
    det.current = detection;
  }, [detection]);

  // Animated regime: a live red<->green equilibrium whose steady state tracks
  // detection. Deps are [tone] only (a stable string literal), so this effect —
  // and therefore the rAF loop and all its buffers — is created exactly once and
  // is NOT recreated on store ticks / SSE events that re-render the parent.
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d", { alpha: true });
    if (!ctx) return;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");

    const st = new Uint8Array(N);
    let grid: Grid | null = null;
    let image: ImageData | null = null;
    let buf: Uint32Array | null = null;

    const sizeCanvas = (): void => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cssWidth = cv.clientWidth || 480;
      grid = buildGrid(cssWidth, dpr);
      cv.width = grid.bw;
      cv.height = grid.bh;
      // ImageData is in device pixels; we write the buffer directly, so no
      // transform is needed (and putImageData ignores transforms anyway).
      image = ctx.createImageData(grid.bw, grid.bh);
      buf = new Uint32Array(image.data.buffer);
    };

    sizeCanvas();

    // Reduced motion: draw a single static snapshot at the detection steady
    // state and stop — no rAF, no CPU burn.
    if (reduced.matches) {
      const d = Math.min(1, Math.max(0, det.current || 0));
      for (let i = 0; i < N; i++) st[i] = Math.random() < d ? 2 : 1;
      if (grid && image && buf) blit(ctx, image, buf, st, grid);
      return;
    }

    let raf = 0;
    let visible = true;

    const tick = (): void => {
      const d = det.current < 0 ? 0 : det.current > 1 ? 1 : det.current || 0;
      const toGreen = d * CHURN;
      const toRed = (1 - d) * CHURN;
      for (let i = 0; i < N; i++) {
        const s = st[i];
        if (s === 0) {
          if (Math.random() < EXPOSE) st[i] = 1;
        } else if (s === 1) {
          if (Math.random() < toGreen) st[i] = 2;
        } else if (Math.random() < toRed) {
          st[i] = 1;
        }
      }
      if (grid && image && buf) blit(ctx, image, buf, st, grid);
      raf = requestAnimationFrame(tick);
    };

    const start = (): void => {
      if (raf) return;
      raf = requestAnimationFrame(tick);
    };
    const stop = (): void => {
      if (!raf) return;
      cancelAnimationFrame(raf);
      raf = 0;
    };

    // Pause the loop while the canvas is scrolled out of view so two 60fps
    // simulations don't burn CPU while the user reads other sections.
    const io = new IntersectionObserver(
      (entries) => {
        visible = entries[0]?.isIntersecting ?? true;
        if (visible) start();
        else stop();
      },
      { threshold: 0 },
    );
    io.observe(cv);

    // Re-fit the backing store + buffers when the container resizes; repaint so
    // the grid stays crisp instead of stretching until the next frame.
    let firstResize = true;
    const ro = new ResizeObserver(() => {
      if (firstResize) {
        firstResize = false;
        return;
      }
      sizeCanvas();
      if (grid && image && buf) blit(ctx, image, buf, st, grid);
    });
    ro.observe(cv);

    if (visible) start();

    return () => {
      stop();
      io.disconnect();
      ro.disconnect();
    };
  }, [tone]);

  return (
    <canvas
      ref={ref}
      style={{ width: "100%", height: CANVAS_H, borderRadius: 12, display: "block" }}
      aria-label={`Customer population — ${tone === "fed" ? "federated" : "siloed"} regime`}
      role="img"
    />
  );
}

// Memoized: the parent (RacePanel) re-renders on every store tick / SSE event /
// attack-banner timer. Without memo, this component's body re-runs each time.
// The only props that matter are detection (read via ref inside the loop) and
// tone (the stable simulation identity), so memo prevents render thrash while
// the ref keeps detection live.
export default memo(PopulationCanvas);
