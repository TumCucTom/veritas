"use client";
import { useEffect, useRef } from "react";

const N = 6000;

// State 0 = unexposed (neutral), 1 = victimised (red), 2 = protected (green).
const COLORS = {
  neutral: "#2a3346",
  atRisk: "#f04a52",
  protected: "#34d399",
} as const;

export default function PopulationCanvas({
  detection,
  tone,
}: {
  detection: number;
  tone: "fed" | "silo";
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const det = useRef(detection);

  // Keep the live detection value in a ref so the rAF loop reads the latest
  // figure without re-subscribing. Synced in an effect (never during render).
  useEffect(() => {
    det.current = detection;
  }, [detection]);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const W = (cv.width = cv.clientWidth || 480);
    const H = (cv.height = 220);
    const cols = 100;
    const rows = Math.ceil(N / cols);
    const st = new Uint8Array(N);
    let raf = 0;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const tick = (): void => {
      const d = Math.min(1, Math.max(0, det.current || 0));
      for (let i = 0; i < N; i++) {
        if (st[i] === 0 && Math.random() < (1 - d) * 0.02) st[i] = 1;
        else if (st[i] === 1 && Math.random() < d * 0.05) st[i] = 2;
      }
      ctx.clearRect(0, 0, W, H);
      const cw = W / cols;
      const ch = H / rows;
      for (let i = 0; i < N; i++) {
        ctx.fillStyle =
          st[i] === 1 ? COLORS.atRisk : st[i] === 2 ? COLORS.protected : COLORS.neutral;
        ctx.fillRect((i % cols) * cw, Math.floor(i / cols) * ch, cw - 1, ch - 1);
      }
      if (!reduce) raf = requestAnimationFrame(tick);
    };

    tick();
    return () => cancelAnimationFrame(raf);
  }, [tone]);

  return (
    <canvas
      ref={ref}
      style={{ width: "100%", height: 220, borderRadius: 12, display: "block" }}
      aria-label={`Customer population — ${tone === "fed" ? "federated" : "siloed"} regime`}
      role="img"
    />
  );
}
