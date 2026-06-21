"use client";

import { memo, useEffect, useMemo, useRef, useState } from "react";

// ── Data schema (web/public/viz/dp.json) ────────────────────────────────────
type EpsPoint = { round: number; epsilon: number };
type TradeoffRow = { sigma: number; epsilon: number; utility: number };

type DPData = {
  delta: number;
  operating: { sigma: number; epsilon: number; round: number };
  epsilonOverRounds: EpsPoint[];
  tradeoff: TradeoffRow[];
  meta: {
    note: string;
    accountant: string;
    sampleRate?: number;
    rounds?: number;
    targetEpsilon?: number;
  };
};

// Tuning ──────────────────────────────────────────────────────────────────
const DRAW_MS = 5200; // time for the eps curve to draw fully left→right
const HOLD_MS = 1800; // dwell on the completed curve before looping

function clamp01(v: number) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
function easeInOut(t: number) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

// Format δ as a compact "1e-5" exponent string.
function formatDelta(d: number): string {
  if (d <= 0) return String(d);
  const exp = Math.round(Math.log10(d));
  const mant = d / Math.pow(10, exp);
  const m = Math.abs(mant - 1) < 1e-9 ? "1" : mant.toFixed(1);
  return `${m}e${exp}`;
}

function PrivacyBudgetInner() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  const [data, setData] = useState<DPData | null>(null);
  const [error, setError] = useState(false);

  // UI-facing readout, synced from the RAF loop at a throttled cadence.
  const [markerRound, setMarkerRound] = useState(0);
  const [markerEps, setMarkerEps] = useState(0);

  // ── Fetch own data ───────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    fetch("/viz/dp.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((json: DPData) => {
        if (cancelled) return;
        if (!json?.epsilonOverRounds?.length || !json?.tradeoff?.length)
          throw new Error("empty");
        setData(json);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // ── Precompute scales + derived headline values ONCE ──────────────────────
  const precomp = useMemo(() => {
    if (!data) return null;
    const eor = data.epsilonOverRounds;
    const tr = data.tradeoff;
    const target = data.meta.targetEpsilon ?? null;

    const maxRound = eor[eor.length - 1].round;
    const maxEpsData = eor[eor.length - 1].epsilon;
    // y-axis tops out a touch above whichever is larger: final eps or target.
    const yTop = Math.max(maxEpsData, target ?? 0) * 1.12;

    // Frontier ranges (eps vs utility), for the small inset chart.
    let epsMin = Infinity;
    let epsMax = -Infinity;
    let utilMin = Infinity;
    let utilMax = -Infinity;
    for (const row of tr) {
      if (row.epsilon < epsMin) epsMin = row.epsilon;
      if (row.epsilon > epsMax) epsMax = row.epsilon;
      if (row.utility < utilMin) utilMin = row.utility;
      if (row.utility > utilMax) utilMax = row.utility;
    }

    // Index of the operating point inside the tradeoff sweep (for the marker).
    let opIdx = 0;
    let bestDiff = Infinity;
    for (let i = 0; i < tr.length; i++) {
      const d = Math.abs(tr[i].sigma - data.operating.sigma);
      if (d < bestDiff) {
        bestDiff = d;
        opIdx = i;
      }
    }

    return {
      eor,
      tr,
      target,
      maxRound,
      yTop,
      epsMin,
      epsMax,
      utilMin,
      utilMax,
      opIdx,
    };
  }, [data]);

  // ── Single RAF loop, held in refs so it isn't recreated on each render ─────
  useEffect(() => {
    if (!data || !precomp) return;
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    const reduceMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    const { eor, tr, target, maxRound, yTop, epsMax, utilMin, utilMax, opIdx } =
      precomp;
    const nPts = eor.length;
    const lastEps = eor[nPts - 1].epsilon;

    let width = 0;
    let height = 0;
    let dpr = 1;

    const sizeCanvas = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = wrap.getBoundingClientRect();
      width = Math.max(320, Math.floor(rect.width));
      height = Math.max(260, Math.floor(rect.height));
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    sizeCanvas();

    let raf = 0;
    let visible = true;
    let animStart = performance.now();

    let lastUiRound = -1;
    let lastUiEps = -1;

    const draw = (now: number) => {
      // ── Layout: main eps-vs-round chart on the left, frontier inset right ──
      const PADL = 52;
      const PADR = 18;
      const PADT = 24;
      const PADB = 40;
      // Reserve right ~30% for the frontier inset on wider panels.
      const insetW = width > 520 ? Math.min(200, width * 0.32) : 0;
      const gap = insetW > 0 ? 26 : 0;
      const mainW = width - insetW - gap;

      const plotL = PADL;
      const plotR = mainW - PADR;
      const plotT = PADT;
      const plotB = height - PADB;
      const plotW = Math.max(1, plotR - plotL);
      const plotH = Math.max(1, plotB - plotT);

      // round → x, epsilon → y (y inverted: more eps = higher = nearer top)
      const rx = (round: number) => plotL + (round / maxRound) * plotW;
      const ey = (eps: number) => plotB - (eps / yTop) * plotH;

      // ── Animation progress (fraction of the curve drawn) ──────────────────
      let progress: number;
      if (reduceMotion) {
        progress = 1; // reduced motion holds the final drawn chart
      } else {
        const cycle = DRAW_MS + HOLD_MS;
        const t = ((now - animStart) % cycle) / cycle;
        const drawFrac = DRAW_MS / cycle;
        progress = t <= drawFrac ? easeInOut(clamp01(t / drawFrac)) : 1;
      }
      // Fractional point index revealed so far.
      const revealF = progress * (nPts - 1);
      const revealIdx = Math.floor(revealF);
      const revealPhase = revealF - revealIdx;

      // ── Background ────────────────────────────────────────────────────────
      ctx.clearRect(0, 0, width, height);
      const bg = ctx.createLinearGradient(0, 0, 0, height);
      bg.addColorStop(0, "rgba(10,14,22,0.96)");
      bg.addColorStop(1, "rgba(14,20,32,0.96)");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, width, height);

      // ── Y grid + axis labels (epsilon) ────────────────────────────────────
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.font = "500 10px ui-sans-serif, system-ui, sans-serif";
      const yTicks = 4;
      for (let i = 0; i <= yTicks; i++) {
        const v = (yTop / yTicks) * i;
        const y = ey(v);
        ctx.strokeStyle = "rgba(120,140,170,0.08)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(plotL, y);
        ctx.lineTo(plotR, y);
        ctx.stroke();
        ctx.fillStyle = "rgba(150,168,196,0.55)";
        ctx.fillText(v.toFixed(1), plotL - 8, y);
      }
      // y axis title
      ctx.save();
      ctx.translate(13, (plotT + plotB) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillStyle = "rgba(150,168,196,0.7)";
      ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText("privacy loss  ε", 0, 0);
      ctx.restore();

      // ── X axis labels (round) ─────────────────────────────────────────────
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = "rgba(150,168,196,0.55)";
      ctx.font = "500 10px ui-sans-serif, system-ui, sans-serif";
      const xTicks = [0, Math.round(maxRound / 2), maxRound];
      for (const r of xTicks) {
        ctx.fillText(String(r), rx(r), plotB + 8);
      }
      ctx.fillStyle = "rgba(150,168,196,0.7)";
      ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText("federated round", (plotL + plotR) / 2, plotB + 22);

      // ── Target-budget line (the budget being spent against) ───────────────
      if (target !== null) {
        const ty = ey(target);
        ctx.setLineDash([5, 4]);
        ctx.strokeStyle = "rgba(244,180,90,0.55)";
        ctx.lineWidth = 1.25;
        ctx.beginPath();
        ctx.moveTo(plotL, ty);
        ctx.lineTo(plotR, ty);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillStyle = "rgba(244,180,90,0.85)";
        ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
        ctx.fillText(`budget  ε = ${target}`, plotL + 4, ty - 3);
      }

      // ── The epsilon curve (drawn up to the reveal point) ──────────────────
      // Build the partial path. Area fill under the curve first, then stroke.
      const segEnd = Math.min(revealIdx, nPts - 1);
      // marker (head) position, interpolated between revealIdx and next point.
      let headX: number;
      let headY: number;
      let headRound: number;
      let headEps: number;
      if (segEnd >= nPts - 1) {
        headX = rx(eor[nPts - 1].round);
        headY = ey(eor[nPts - 1].epsilon);
        headRound = eor[nPts - 1].round;
        headEps = eor[nPts - 1].epsilon;
      } else {
        const a = eor[segEnd];
        const b = eor[segEnd + 1];
        headRound = a.round + (b.round - a.round) * revealPhase;
        headEps = a.epsilon + (b.epsilon - a.epsilon) * revealPhase;
        headX = rx(headRound);
        headY = ey(headEps);
      }

      // area fill
      const grad = ctx.createLinearGradient(0, plotT, 0, plotB);
      grad.addColorStop(0, "rgba(96,165,250,0.26)");
      grad.addColorStop(1, "rgba(96,165,250,0.02)");
      ctx.beginPath();
      ctx.moveTo(rx(eor[0].round), plotB);
      ctx.lineTo(rx(eor[0].round), ey(eor[0].epsilon));
      for (let i = 1; i <= segEnd; i++) {
        ctx.lineTo(rx(eor[i].round), ey(eor[i].epsilon));
      }
      ctx.lineTo(headX, headY);
      ctx.lineTo(headX, plotB);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();

      // stroke line
      ctx.beginPath();
      ctx.moveTo(rx(eor[0].round), ey(eor[0].epsilon));
      for (let i = 1; i <= segEnd; i++) {
        ctx.lineTo(rx(eor[i].round), ey(eor[i].epsilon));
      }
      ctx.lineTo(headX, headY);
      ctx.strokeStyle = "rgba(122,180,255,0.95)";
      ctx.lineWidth = 2.2;
      ctx.lineJoin = "round";
      ctx.stroke();

      // ── Marker riding the curve head ──────────────────────────────────────
      // soft glow
      const glowR = 14;
      const g = ctx.createRadialGradient(headX, headY, 0, headX, headY, glowR);
      g.addColorStop(0, "rgba(122,180,255,0.5)");
      g.addColorStop(1, "rgba(122,180,255,0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(headX, headY, glowR, 0, Math.PI * 2);
      ctx.fill();
      // dot
      ctx.fillStyle = "rgb(180,212,255)";
      ctx.beginPath();
      ctx.arc(headX, headY, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "rgba(10,14,22,0.9)";
      ctx.stroke();

      // floating "round N → ε" tag near the marker
      const tag = `round ${Math.round(headRound)} → ε ${headEps.toFixed(2)}`;
      ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
      const tagW = ctx.measureText(tag).width + 14;
      let tagX = headX + 10;
      if (tagX + tagW > plotR) tagX = headX - 10 - tagW;
      const tagY = Math.max(plotT + 2, headY - 26);
      ctx.fillStyle = "rgba(16,24,40,0.92)";
      ctx.strokeStyle = "rgba(122,180,255,0.4)";
      ctx.lineWidth = 1;
      roundRect(ctx, tagX, tagY, tagW, 19, 5);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "rgb(196,220,255)";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(tag, tagX + 7, tagY + 10);

      // ── Frontier inset: ε vs utility, with operating point marked ─────────
      if (insetW > 0) {
        const ix0 = mainW + gap;
        const iy0 = PADT;
        const iw = insetW - 8;
        const ih = plotH;
        const ix1 = ix0 + iw;
        const iy1 = iy0 + ih;

        // frame
        ctx.strokeStyle = "rgba(120,140,170,0.14)";
        ctx.lineWidth = 1;
        ctx.strokeRect(ix0, iy0, iw, ih);

        // title
        ctx.fillStyle = "rgba(176,196,224,0.8)";
        ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText("noise ↔ utility frontier", ix0, iy0 - 16);

        // scales: x = epsilon (0..epsMax), y = utility (utilMin..utilMax)
        const ux = (e: number) => ix0 + (e / epsMax) * iw;
        const uSpan = Math.max(1e-6, utilMax - utilMin);
        const uy = (u: number) =>
          iy1 - ((u - utilMin) / uSpan) * (ih - 14) - 7;

        // frontier curve
        ctx.beginPath();
        for (let i = 0; i < tr.length; i++) {
          const x = ux(tr[i].epsilon);
          const y = uy(tr[i].utility);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = "rgba(140,200,170,0.85)";
        ctx.lineWidth = 1.8;
        ctx.lineJoin = "round";
        ctx.stroke();

        // points
        for (let i = 0; i < tr.length; i++) {
          const x = ux(tr[i].epsilon);
          const y = uy(tr[i].utility);
          ctx.fillStyle = "rgba(140,200,170,0.55)";
          ctx.beginPath();
          ctx.arc(x, y, 2, 0, Math.PI * 2);
          ctx.fill();
        }

        // operating point: ringed amber dot + crosshair
        const ox = ux(tr[opIdx].epsilon);
        const oy = uy(tr[opIdx].utility);
        ctx.strokeStyle = "rgba(244,180,90,0.35)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ix0, oy);
        ctx.lineTo(ix1, oy);
        ctx.moveTo(ox, iy0);
        ctx.lineTo(ox, iy1);
        ctx.stroke();
        ctx.fillStyle = "rgb(244,188,104)";
        ctx.beginPath();
        ctx.arc(ox, oy, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "rgba(10,14,22,0.9)";
        ctx.stroke();

        // axis hints
        ctx.fillStyle = "rgba(150,168,196,0.6)";
        ctx.font = "500 9px ui-sans-serif, system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.fillText("ε →", ix0 + 2, iy1 + 12);
        ctx.save();
        ctx.translate(ix1 + 10, (iy0 + iy1) / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText("recall →", 0, 0);
        ctx.restore();

        // operating label
        ctx.fillStyle = "rgba(244,188,104,0.9)";
        ctx.font = "600 9px ui-sans-serif, system-ui, sans-serif";
        ctx.textAlign = ox > ix0 + iw * 0.6 ? "right" : "left";
        ctx.textBaseline = "bottom";
        const olx = ox > ix0 + iw * 0.6 ? ox - 6 : ox + 6;
        ctx.fillText("chosen σ", olx, oy - 6);
      }

      // ── Throttled UI sync ────────────────────────────────────────────────
      const uiRound = Math.round(headRound);
      if (uiRound !== lastUiRound) {
        lastUiRound = uiRound;
        setMarkerRound(uiRound);
      }
      const uiEps = Math.round(headEps * 100);
      if (uiEps !== lastUiEps) {
        lastUiEps = uiEps;
        setMarkerEps(headEps);
      }

      if (visible) raf = requestAnimationFrame(draw);
    };

    const start = () => {
      cancelAnimationFrame(raf);
      if (!reduceMotion) animStart = performance.now();
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
    // silence unused-var lints for values only read in some branches
    void lastEps;

    return () => {
      stop();
      io.disconnect();
      ro.disconnect();
    };
  }, [data, precomp]);

  const delta = data?.delta ?? 1e-5;
  const opEps = data?.operating.epsilon ?? 0;
  const opSigma = data?.operating.sigma ?? 0;
  const target = data?.meta.targetEpsilon ?? null;

  // ── Fallback ────────────────────────────────────────────────────────────
  if (error) {
    return (
      <section
        aria-label="Differential-privacy budget"
        className="rise overflow-hidden rounded-[20px] border bg-bg-surface/80 p-6 backdrop-blur-sm sm:p-8"
        style={{
          borderColor: "var(--border-default)",
          boxShadow: "var(--shadow-panel)",
        }}
      >
        <p className="eyebrow text-accent-gold">
          Differential privacy · RDP accountant
        </p>
        <h2 className="mt-2 font-display text-[clamp(1.5rem,1.1rem+1.6vw,2.4rem)] leading-tight tracking-tight text-text-primary">
          What &ldquo;private&rdquo; actually costs.
        </h2>
        <p className="mt-4 text-[13px] leading-relaxed text-text-secondary">
          The privacy-accounting feed is unavailable right now. Veritas still
          tracks every federated round in Rényi-DP space and converts to a
          rigorous (ε, δ) budget — this dashboard will resume once the data feed
          returns.
        </p>
      </section>
    );
  }

  return (
    <section
      aria-labelledby="dp-budget-heading"
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
            Differential privacy · RDP accountant
          </p>
          <h2
            id="dp-budget-heading"
            className="mt-2 font-display text-[clamp(1.7rem,1.2rem+2vw,3rem)] leading-tight tracking-tight text-text-primary"
          >
            What &ldquo;private&rdquo; actually costs.
          </h2>
          <p className="mt-3 max-w-3xl text-[13px] leading-relaxed text-text-secondary sm:text-[14px]">
            Every federated round adds calibrated Gaussian noise. A rigorous
            Rényi-DP accountant composes those rounds and converts them to a real
            (ε, δ) privacy budget — so the leakage is measured, not assumed.
          </p>
        </div>
        <div
          className="grid grid-cols-2 gap-px overflow-hidden rounded-[14px] border text-center sm:min-w-[300px]"
          style={{
            borderColor: "var(--border-default)",
            background: "var(--border-default)",
          }}
        >
          <Stat
            label={`ε spent · δ=${formatDelta(delta)}`}
            value={opEps ? opEps.toFixed(2) : "—"}
            tone="fed"
          />
          <Stat
            label="noise σ"
            value={opSigma ? opSigma.toFixed(2) : "—"}
            tone="gold"
          />
        </div>
      </header>

      <div className="p-4 sm:p-5">
        <div
          ref={wrapRef}
          className="relative h-[300px] w-full overflow-hidden rounded-[14px] border bg-bg-deep sm:h-[340px]"
          style={{ borderColor: "var(--border-default)" }}
        >
          <canvas
            ref={canvasRef}
            className="block h-full w-full"
            role="img"
            aria-label="Line chart of privacy loss epsilon rising over federated rounds toward the target budget, with a noise-versus-utility frontier inset"
          />

          {/* Live marker readout overlay */}
          <div
            className="pointer-events-none absolute right-3 top-3 flex flex-col items-end gap-0.5 rounded-[10px] border px-3 py-1.5 text-right backdrop-blur-sm"
            style={{
              borderColor: "var(--border-default)",
              background: "rgba(12,18,30,0.7)",
            }}
          >
            <span className="eyebrow" style={{ color: "var(--fed)" }}>
              round {String(markerRound).padStart(2, "0")}
            </span>
            <span className="tabular font-display text-lg leading-none text-text-primary">
              ε {markerEps.toFixed(2)}
            </span>
          </div>
        </div>

        {/* Caption — the guarantee in one plain sentence */}
        <p className="mt-3 text-[12px] leading-relaxed text-text-secondary sm:text-[13px]">
          After {data?.operating.round ?? 30} rounds the model leaks at most ε ≈{" "}
          {opEps ? opEps.toFixed(2) : "—"} at δ = {formatDelta(delta)}
          {target !== null ? `, comfortably under the ε = ${target} budget` : ""}:
          no attacker — even one who knows everyone else&rsquo;s data — can tell
          whether any single record was used by more than an e
          <sup className="text-[9px]">ε</sup> factor. A higher σ buys a smaller ε
          at a measured cost in recall; we pick a calibrated point on that
          frontier.
        </p>

        {/* Legend */}
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1.5">
          <LegendRow color="rgba(122,180,255,0.95)" line label="ε spent (RDP)" />
          {target !== null && (
            <LegendRow color="rgba(244,180,90,0.85)" dash label="target budget" />
          )}
          <LegendRow
            color="rgba(140,200,170,0.85)"
            line
            label="noise↔utility frontier"
          />
          <LegendRow color="rgb(244,188,104)" label="chosen operating σ" />
          <p className="tabular ml-auto text-[12px] text-text-muted">
            {data?.meta.accountant ?? "RDP"} accountant · subsampled Gaussian
          </p>
        </div>
      </div>
    </section>
  );
}

// ── canvas helper: rounded rect path ───────────────────────────────────────
function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
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
      <p className="tabular mt-1 font-display text-2xl leading-none text-text-primary">
        {value}
      </p>
    </div>
  );
}

function LegendRow({
  color,
  label,
  line,
  dash,
}: {
  color: string;
  label: string;
  line?: boolean;
  dash?: boolean;
}) {
  return (
    <span className="flex items-center gap-2 text-[11px] text-text-secondary">
      {line || dash ? (
        <span
          aria-hidden
          className="inline-block h-[2px] w-4 rounded-full"
          style={{
            background: dash
              ? `repeating-linear-gradient(to right, ${color} 0 4px, transparent 4px 7px)`
              : color,
          }}
        />
      ) : (
        <span
          aria-hidden
          className="inline-block h-2.5 w-2.5 rounded-full"
          style={{ background: color }}
        />
      )}
      {label}
    </span>
  );
}

const PrivacyBudget = memo(PrivacyBudgetInner);
export default PrivacyBudget;
