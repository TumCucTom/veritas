"use client";
import { useEffect, useRef } from "react";

// Number of dots painted per regime panel. Exported so callers can derive an
// honest "customers per dot" caption from the live customer total instead of
// inventing a ratio.
export const POPULATION_DOTS = 6000;
const N = POPULATION_DOTS;
const COLS = 100;
const ROWS = Math.ceil(N / COLS);
const CANVAS_H = 220;

// State 0 = unexposed (neutral), 1 = victimised (red), 2 = protected (green).
const COLORS = {
  neutral: "#2a3346",
  atRisk: "#f04a52",
  protected: "#34d399",
} as const;

// Per-frame transition rates. The red<->green equilibrium is tuned so the
// steady-state share of *protected* dots equals the regime's detection rate:
// red->green ∝ detection, green->red ∝ (1 - detection). The stationary green
// fraction is therefore exactly `detection`, so the siloed panel keeps a
// visible red band (low detection) while the federated panel clears to green.
// That persistent gap is the whole story — without the relapse term both
// regimes would simply accumulate green over time and the contrast would wash out.
const EXPOSE = 0.025; // neutral -> at-risk as the campaign reaches customers
const CHURN = 0.06; // base rate of the protection dynamics

function paint(
  ctx: CanvasRenderingContext2D,
  st: Uint8Array,
  width: number,
  height: number,
): void {
  ctx.clearRect(0, 0, width, height);
  const cw = width / COLS;
  const ch = height / ROWS;
  for (let i = 0; i < N; i++) {
    ctx.fillStyle =
      st[i] === 1 ? COLORS.atRisk : st[i] === 2 ? COLORS.protected : COLORS.neutral;
    ctx.fillRect((i % COLS) * cw, Math.floor(i / COLS) * ch, cw - 1, ch - 1);
  }
}

type Tone = "fed" | "silo";

interface PopulationCanvasProps {
  detection: number;
  tone: Tone;
}

export default function PopulationCanvas({ detection, tone }: PopulationCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const det = useRef(detection);

  // Keep the live detection value in a ref so the rAF loop reads the latest
  // figure without restarting the simulation. Synced in an effect (never during render).
  useEffect(() => {
    det.current = detection;
  }, [detection]);

  // Animated regime: a live red<->green equilibrium whose steady state tracks detection.
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const width = (cv.width = cv.clientWidth || 480);
    const height = (cv.height = CANVAS_H);
    const st = new Uint8Array(N);
    let raf = 0;

    const tick = (): void => {
      const d = Math.min(1, Math.max(0, det.current || 0));
      for (let i = 0; i < N; i++) {
        const s = st[i];
        if (s === 0) {
          if (Math.random() < EXPOSE) st[i] = 1;
        } else if (s === 1) {
          if (Math.random() < d * CHURN) st[i] = 2;
        } else if (Math.random() < (1 - d) * CHURN) {
          st[i] = 1;
        }
      }
      paint(ctx, st, width, height);
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, [tone]);

  // Reduced-motion: draw a single static snapshot at the detection steady state.
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const width = (cv.width = cv.clientWidth || 480);
    const height = (cv.height = CANVAS_H);
    const d = Math.min(1, Math.max(0, detection || 0));
    const st = new Uint8Array(N);
    for (let i = 0; i < N; i++) st[i] = Math.random() < d ? 2 : 1;
    paint(ctx, st, width, height);
  }, [tone, detection]);

  return (
    <canvas
      ref={ref}
      style={{ width: "100%", height: CANVAS_H, borderRadius: 12, display: "block" }}
      aria-label={`Customer population — ${tone === "fed" ? "federated" : "siloed"} regime`}
      role="img"
    />
  );
}
