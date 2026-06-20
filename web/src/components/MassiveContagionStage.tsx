"use client";
import { useEffect, useMemo, useRef } from "react";
import { meanDetection } from "../lib/derive";
import { formatNumber } from "../lib/format";
import { useVeritas } from "../lib/store";
import {
  LOGICAL_CUSTOMERS,
  computeContagionFrame,
  type ContagionFrame,
  type ContagionRegime,
} from "../lib/contagion";

const COLORS = {
  neutral: "rgba(82, 97, 128, 0.34)",
  exposed: "rgba(240, 74, 82, 0.86)",
  protected: "rgba(52, 211, 153, 0.9)",
  corridor: "rgba(214, 168, 91, 0.18)",
  band: "rgba(174, 182, 198, 0.06)",
} as const;

export default function MassiveContagionStage() {
  const { state } = useVeritas();
  const banks = state?.banks ?? [];
  const round = state?.round ?? 0;
  const campaignActive = state?.campaignActive ?? false;

  const siloDetection = meanDetection(banks, "siloed");
  const fedDetection = meanDetection(banks, "federated");

  const siloed = useMemo(
    () =>
      computeContagionFrame({
        regime: "siloed",
        round,
        campaignActive,
        detection: siloDetection,
      }),
    [campaignActive, round, siloDetection],
  );
  const federated = useMemo(
    () =>
      computeContagionFrame({
        regime: "federated",
        round,
        campaignActive,
        detection: fedDetection,
      }),
    [campaignActive, fedDetection, round],
  );

  return (
    <section
      aria-labelledby="contagion-stage-heading"
      className="rise overflow-hidden rounded-[20px] border bg-bg-surface/80 backdrop-blur-sm"
      style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-panel)" }}
    >
      <header className="grid gap-4 border-b p-5 sm:p-6 lg:grid-cols-[1fr_auto] lg:items-end"
        style={{ borderColor: "var(--border-default)" }}
      >
        <div>
          <p className="eyebrow text-accent-gold">Million-customer contagion model</p>
          <h2
            id="contagion-stage-heading"
            className="mt-2 font-display text-[clamp(1.7rem,1.2rem+2vw,3rem)] leading-tight tracking-tight text-text-primary"
          >
            Fraud spreads through the network. Veritas spreads faster.
          </h2>
          <p className="mt-3 max-w-3xl text-[13px] leading-relaxed text-text-secondary sm:text-[14px]">
            A deterministic SIR-style graph simulation over {formatNumber(LOGICAL_CUSTOMERS)}{" "}
            simulated customers. Bank clusters are local communities; bright corridors are sampled
            mule-payment paths between institutions.
          </p>
        </div>
        <div
          className="grid grid-cols-2 gap-px overflow-hidden rounded-[14px] border text-center sm:min-w-[360px]"
          style={{ borderColor: "var(--border-default)", background: "var(--border-default)" }}
        >
          <Metric label="Siloed exposed" value={siloed.totals.exposed} tone="silo" />
          <Metric label="Suppressed by Veritas" value={federated.totals.suppressed} tone="fed" />
        </div>
      </header>

      <div className="grid gap-px bg-border-default lg:grid-cols-2">
        <ContagionPanel
          title="Siloed"
          label="Fraud storm"
          regime="siloed"
          frame={siloed}
          round={round}
        />
        <ContagionPanel
          title="Veritas"
          label="Immunity wave"
          regime="federated"
          frame={federated}
          round={round}
        />
      </div>
    </section>
  );
}

function Metric({ label, value, tone }: { label: string; value: number; tone: "fed" | "silo" }) {
  return (
    <div className="bg-bg-surface-2 px-4 py-3">
      <p className="eyebrow" style={{ color: tone === "fed" ? "var(--fed)" : "var(--silo)" }}>
        {label}
      </p>
      <p className="tabular mt-1 font-display text-2xl leading-none text-text-primary">
        {formatNumber(value)}
      </p>
    </div>
  );
}

function ContagionPanel({
  title,
  label,
  regime,
  frame,
  round,
}: {
  title: string;
  label: string;
  regime: ContagionRegime;
  frame: ContagionFrame;
  round: number;
}) {
  return (
    <article className="bg-bg-surface p-4 sm:p-5">
      <div className="mb-3 flex items-start justify-between gap-4">
        <div>
          <p className="eyebrow" style={{ color: regime === "federated" ? "var(--fed)" : "var(--silo)" }}>
            {label}
          </p>
          <h3 className="font-display text-2xl tracking-tight text-text-primary">{title}</h3>
        </div>
        <div className="text-right">
          <p className="eyebrow">round</p>
          <p className="tabular font-display text-2xl leading-none text-text-primary">
            {String(round).padStart(2, "0")}
          </p>
        </div>
      </div>

      <ContagionCanvas frame={frame} regime={regime} round={round} />

      <dl className="mt-3 grid grid-cols-3 gap-3">
        <SmallStat label="exposed" value={frame.totals.exposed} tone="silo" />
        <SmallStat label="protected" value={frame.totals.protected} tone="fed" />
        <SmallStat label="mule links" value={frame.totals.crossBankLinks} tone="gold" />
      </dl>
    </article>
  );
}

function SmallStat({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: "fed" | "silo" | "gold";
}) {
  const color = tone === "fed" ? "var(--fed)" : tone === "silo" ? "var(--silo)" : "var(--accent-gold)";
  return (
    <div>
      <dt className="eyebrow mb-1">{label}</dt>
      <dd className="tabular truncate font-display text-xl leading-none" style={{ color }}>
        {formatNumber(value)}
      </dd>
    </div>
  );
}

function ContagionCanvas({
  frame,
  regime,
  round,
}: {
  frame: ContagionFrame;
  regime: ContagionRegime;
  round: number;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(520, Math.floor(rect.width));
    const height = Math.max(320, Math.floor(rect.height));
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    let raf = 0;
    const started = performance.now();
    const duration = media.matches ? 1 : 820;

    const tick = (now: number) => {
      const elapsed = Math.max(0, now - started);
      const phase = duration === 1 ? 1 : Math.min(1, elapsed / duration);
      drawFrame(ctx, frame, regime, width, height, phase);
      if (phase < 1) raf = requestAnimationFrame(tick);
    };

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [frame, regime, round]);

  return (
    <canvas
      ref={ref}
      className="block h-[320px] w-full rounded-[14px] border bg-bg-deep sm:h-[380px]"
      style={{ borderColor: "var(--border-default)" }}
      role="img"
      aria-label={`${regime === "federated" ? "Veritas" : "Siloed"} million-customer contagion graph`}
    />
  );
}

function drawFrame(
  ctx: CanvasRenderingContext2D,
  frame: ContagionFrame,
  regime: ContagionRegime,
  width: number,
  height: number,
  phase: number,
) {
  ctx.clearRect(0, 0, width, height);
  const bg = ctx.createLinearGradient(0, 0, width, height);
  bg.addColorStop(0, "rgba(10,14,22,0.98)");
  bg.addColorStop(0.55, "rgba(13,19,32,0.98)");
  bg.addColorStop(1, regime === "federated" ? "rgba(13,110,79,0.16)" : "rgba(122,29,34,0.18)");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  drawCorridors(ctx, frame, width, height);
  drawBands(ctx, frame, width, height);

  const wave = regime === "federated" ? easeOut(phase) : Math.min(1, phase * 1.15);
  const pointSize = width > 620 ? 1.35 : 1.05;
  for (const point of frame.points) {
    let state = point.state;
    const centerDistance = Math.hypot(point.x - 0.5, point.y - 0.5);
    const sourceDistance = Math.min(Math.hypot(point.x - 0.14, point.y - 0.2), Math.hypot(point.x - 0.83, point.y - 0.78));
    if (state === "protected" && centerDistance > wave * 0.78 + 0.08) state = "neutral";
    if (state === "exposed" && sourceDistance > wave * 0.88 + 0.08) state = "neutral";

    ctx.globalAlpha = state === "neutral" ? 0.52 : 0.58 + point.intensity * 0.42;
    ctx.fillStyle =
      state === "protected" ? COLORS.protected : state === "exposed" ? COLORS.exposed : COLORS.neutral;
    ctx.fillRect(point.x * width, point.y * height, point.hub ? pointSize * 2.4 : pointSize, point.hub ? pointSize * 2.4 : pointSize);
  }
  ctx.globalAlpha = 1;

  drawWave(ctx, regime, width, height, wave);
}

function drawBands(ctx: CanvasRenderingContext2D, frame: ContagionFrame, width: number, height: number) {
  for (const band of frame.bands) {
    const r = band.radius * Math.min(width, height) * 1.12;
    ctx.beginPath();
    ctx.arc(band.cx * width, band.cy * height, r, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.band;
    ctx.fill();
    ctx.strokeStyle = "rgba(174, 182, 198, 0.08)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

function drawCorridors(ctx: CanvasRenderingContext2D, frame: ContagionFrame, width: number, height: number) {
  ctx.save();
  ctx.strokeStyle = COLORS.corridor;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 7]);
  for (const band of frame.bands) {
    const target = frame.bands[(band.bank + 3) % frame.bands.length];
    ctx.beginPath();
    ctx.moveTo(band.cx * width, band.cy * height);
    ctx.lineTo(target.cx * width, target.cy * height);
    ctx.stroke();
  }
  ctx.restore();
}

function drawWave(
  ctx: CanvasRenderingContext2D,
  regime: ContagionRegime,
  width: number,
  height: number,
  phase: number,
) {
  const x = regime === "federated" ? width / 2 : width * 0.16;
  const y = regime === "federated" ? height / 2 : height * 0.2;
  const radius = phase * Math.max(width, height) * (regime === "federated" ? 0.62 : 0.48);
  const gradient = ctx.createRadialGradient(x, y, Math.max(1, radius * 0.62), x, y, Math.max(2, radius));
  const color = regime === "federated" ? "52, 211, 153" : "240, 74, 82";
  gradient.addColorStop(0, `rgba(${color}, 0)`);
  gradient.addColorStop(0.72, `rgba(${color}, 0.02)`);
  gradient.addColorStop(1, `rgba(${color}, 0.26)`);
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 18;
  ctx.stroke();
}

function easeOut(value: number): number {
  return 1 - Math.pow(1 - value, 3);
}
