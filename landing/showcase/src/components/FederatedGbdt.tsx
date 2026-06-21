"use client";

import { memo, useEffect, useMemo, useRef, useState } from "react";

// ── Data schema (web/public/viz/gbdt.json) ─────────────────────────────────
type BankHistogram = {
  bank: number;
  counts: number[];
};

type TreeNode = {
  id: number;
  feature: string;
  threshold: number;
  parent: number | null;
  side: "L" | "R" | null;
};

type GbdtData = {
  feature: string;
  bins: number[];
  bankHistograms: BankHistogram[];
  globalHistogram: number[];
  split: { binIndex: number; gain: number };
  tree: TreeNode[];
  meta: { note: string; model: string; metric?: number };
};

// ── Animation stages ────────────────────────────────────────────────────────
// 0..nBanks-1 : add bank b into the global histogram (one per step)
// nBanks      : global complete, draw the split line + gain
// nBanks+1    : grow the resulting tree node(s)
const ADD_MS = 1100; // time to stack one bank in
const SPLIT_MS = 1700; // dwell while the split line sweeps in
const TREE_MS = 2600; // dwell while the tree forms + final hold

function clamp01(v: number) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
function easeOut(t: number) {
  return 1 - Math.pow(1 - t, 3);
}

// Site palette (mirrors MuleGraph tokens): cool slate for banks, warm gold for
// the aggregate + the chosen split.
const BANK_HUES = [205, 188, 172, 220]; // cool teal/slate spread per bank

function FederatedGbdtInner() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  const [data, setData] = useState<GbdtData | null>(null);
  const [error, setError] = useState(false);

  // Stage index drives the caption (UI state, throttled from the RAF loop).
  const [stage, setStage] = useState(0);
  const [banksIn, setBanksIn] = useState(0);

  // ── Fetch own data ───────────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    fetch("viz/gbdt.json")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((json: GbdtData) => {
        if (cancelled) return;
        if (!json?.bankHistograms?.length || !json?.globalHistogram?.length)
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

  // ── Precompute derived constants ONCE ────────────────────────────────────
  const precomp = useMemo(() => {
    if (!data) return null;
    const nBanks = data.bankHistograms.length;
    const nBins = data.globalHistogram.length;
    const globalMax = Math.max(1, ...data.globalHistogram);
    const bankMax = Math.max(
      1,
      ...data.bankHistograms.flatMap((h) => h.counts),
    );
    // Reusable accumulation buffer (allocated once, never per-frame).
    const accum = new Float32Array(nBins);
    return { nBanks, nBins, globalMax, bankMax, accum };
  }, [data]);

  const captions = useMemo(() => {
    if (!data || !precomp) return [] as string[];
    const out: string[] = [];
    for (let b = 0; b < precomp.nBanks; b++) {
      out.push(
        `Bank ${b + 1} sends its “${data.feature}” histogram — counts per bin, never customer rows.`,
      );
    }
    out.push(
      "Securely summed: the global histogram is the masked sum of every bank’s counts.",
    );
    out.push(
      `Best-gain split chosen on the global view (gain ${data.split.gain.toFixed(1)}) — the tree grows.`,
    );
    return out;
  }, [data, precomp]);

  // ── Single RAF render loop, held in refs ─────────────────────────────────
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

    const { nBanks, nBins, globalMax, bankMax, accum } = precomp;
    const banks = data.bankHistograms;
    const global = data.globalHistogram;
    const splitBin = data.split.binIndex;
    const tree = data.tree;
    const lastStage = nBanks + 1; // split stage = nBanks, tree stage = nBanks+1
    const totalStages = nBanks + 2;

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
    let stageStart = performance.now();
    // reduced-motion holds the FINAL frame (everything summed, split + tree).
    let curStage = reduceMotion ? lastStage : 0;
    let lastUiStage = -1;
    let lastUiBanksIn = -1;

    const stageDuration = (s: number) => {
      if (s < nBanks) return ADD_MS;
      if (s === nBanks) return SPLIT_MS;
      return TREE_MS;
    };

    const draw = (now: number) => {
      // ── Advance the stage clock (unless reduced motion / offscreen) ───────
      let phase: number;
      if (reduceMotion) {
        phase = 1;
        stageStart = now;
      } else {
        const dur = stageDuration(curStage);
        const raw = (now - stageStart) / dur;
        if (raw >= 1) {
          curStage = curStage >= lastStage ? 0 : curStage + 1;
          stageStart = now;
          phase = 0;
        } else {
          phase = clamp01(raw);
        }
      }

      // Banks fully summed-in count, plus the fractional bank currently arriving.
      const fullBanks = Math.min(curStage, nBanks);
      const arriving = curStage < nBanks ? easeOut(phase) : 0;
      const banksInNow = Math.min(nBanks, fullBanks + (arriving > 0 ? 1 : 0));

      // ── Layout: bank row across the top, global histogram below ───────────
      const PAD = 22;
      const W = width - PAD * 2;
      const topH = Math.round(height * 0.32);
      const gapY = 18;
      const globalTop = PAD + topH + gapY + 14;
      const globalH = Math.round(height * 0.30);
      const treeTop = globalTop + globalH + 26;

      ctx.clearRect(0, 0, width, height);
      const bg = ctx.createLinearGradient(0, 0, width, height);
      bg.addColorStop(0, "rgba(8,12,20,0.96)");
      bg.addColorStop(0.6, "rgba(11,16,28,0.96)");
      bg.addColorStop(1, "rgba(16,23,38,0.96)");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, width, height);

      // ── Bank mini-histograms (top row) ────────────────────────────────────
      const cellGap = 14;
      const cellW = (W - cellGap * (nBanks - 1)) / nBanks;
      const miniH = topH - 16;
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      for (let b = 0; b < nBanks; b++) {
        const x0 = PAD + b * (cellW + cellGap);
        const y0 = PAD + 4;
        const summed = b < fullBanks;
        const isArriving = b === fullBanks && arriving > 0;
        // dim once a bank has been folded in (its counts now live in global)
        const alpha = summed ? 0.28 : isArriving ? 0.45 + arriving * 0.45 : 0.9;
        const hue = BANK_HUES[b % BANK_HUES.length];

        ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
        ctx.fillStyle = `hsla(${hue},60%,72%,${(alpha * 0.85).toFixed(3)})`;
        ctx.fillText(`Bank ${b + 1}`, x0, y0 + 9);

        const bars = banks[b].counts;
        const bw = (cellW - (nBins - 1) * 1.5) / nBins;
        const baseY = y0 + 14 + miniH;
        for (let i = 0; i < nBins; i++) {
          const h = (bars[i] / bankMax) * miniH;
          const bx = x0 + i * (bw + 1.5);
          ctx.fillStyle = `hsla(${hue},55%,58%,${alpha.toFixed(3)})`;
          ctx.fillRect(bx, baseY - h, Math.max(1, bw), h);
        }
        // a faint flow tick when arriving
        if (isArriving) {
          ctx.fillStyle = `hsla(${hue},70%,66%,${(0.5 * arriving).toFixed(3)})`;
          ctx.fillRect(x0, baseY + 4, cellW, 2);
        }
      }

      // ── Build the partial global histogram for this moment ────────────────
      // Sum the fully-folded banks, then add the fractional arriving bank.
      for (let i = 0; i < nBins; i++) accum[i] = 0;
      for (let b = 0; b < fullBanks; b++) {
        const c = banks[b].counts;
        for (let i = 0; i < nBins; i++) accum[i] += c[i];
      }
      if (arriving > 0 && fullBanks < nBanks) {
        const c = banks[fullBanks].counts;
        for (let i = 0; i < nBins; i++) accum[i] += c[i] * arriving;
      }

      // ── Global histogram (the securely-summed aggregate) ──────────────────
      ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
      ctx.fillStyle = "rgba(246,233,205,0.75)";
      ctx.textAlign = "left";
      ctx.fillText(
        `Global histogram · secure sum of bank counts · feature “${data.feature}”`,
        PAD,
        globalTop - 6,
      );

      const gBaseY = globalTop + globalH;
      const gbw = (W - (nBins - 1) * 3) / nBins;
      for (let i = 0; i < nBins; i++) {
        const h = (accum[i] / globalMax) * globalH;
        const bx = PAD + i * (gbw + 3);
        const isSplitSide = curStage >= nBanks && i > splitBin;
        // warm gold bars; right-of-split tinted slightly differently once split
        const hue = isSplitSide ? 28 : 42;
        const grad = ctx.createLinearGradient(0, gBaseY - h, 0, gBaseY);
        grad.addColorStop(0, `hsla(${hue},92%,62%,0.92)`);
        grad.addColorStop(1, `hsla(${hue},88%,48%,0.55)`);
        ctx.fillStyle = grad;
        ctx.fillRect(bx, gBaseY - h, Math.max(1, gbw), h);
      }
      // baseline
      ctx.strokeStyle = "rgba(150,170,200,0.18)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(PAD, gBaseY + 0.5);
      ctx.lineTo(PAD + W, gBaseY + 0.5);
      ctx.stroke();

      // ── Split line + gain (stage >= nBanks) ───────────────────────────────
      if (curStage >= nBanks) {
        const splitPhase = curStage === nBanks ? easeOut(phase) : 1;
        const sx = PAD + (splitBin + 1) * (gbw + 3) - 1.5;
        const lineTop = globalTop - 2;
        const lineBot = gBaseY + 6;
        const drawBot = lineTop + (lineBot - lineTop) * splitPhase;
        ctx.strokeStyle = "rgba(246,233,205,0.92)";
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(sx, lineTop);
        ctx.lineTo(sx, drawBot);
        ctx.stroke();
        ctx.setLineDash([]);
        if (splitPhase > 0.6) {
          const a = clamp01((splitPhase - 0.6) / 0.4);
          ctx.fillStyle = `rgba(246,233,205,${a.toFixed(3)})`;
          ctx.font = "700 11px ui-sans-serif, system-ui, sans-serif";
          ctx.textAlign = sx > PAD + W * 0.7 ? "right" : "left";
          const lx = sx > PAD + W * 0.7 ? sx - 6 : sx + 6;
          ctx.fillText(`split · gain ${data.split.gain.toFixed(1)}`, lx, lineTop + 12);
        }
      }

      // ── Resulting tree (final stage) ──────────────────────────────────────
      if (curStage >= lastStage && tree.length) {
        const treePhase = curStage === lastStage ? easeOut(phase) : 1;
        drawTree(ctx, tree, PAD, treeTop, W, height - treeTop - 10, treePhase);
      }

      // ── Throttled UI sync ────────────────────────────────────────────────
      if (curStage !== lastUiStage) {
        lastUiStage = curStage;
        setStage(curStage);
      }
      if (banksInNow !== lastUiBanksIn) {
        lastUiBanksIn = banksInNow;
        setBanksIn(banksInNow);
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

    return () => {
      stop();
      io.disconnect();
      ro.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- totalStages/lastStage derive from data
  }, [data, precomp]);

  const caption =
    captions.length && stage < captions.length
      ? captions[stage]
      : (captions[captions.length - 1] ?? "");

  // ── Fallback ──────────────────────────────────────────────────────────────
  if (error) {
    return (
      <section
        aria-label="Federated GBDT histograms"
        className="rise overflow-hidden rounded-[20px] border bg-bg-surface/80 p-6 backdrop-blur-sm sm:p-8"
        style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-panel)" }}
      >
        <p className="eyebrow text-accent-gold">Federated GBDT · histograms, not data</p>
        <h2 className="mt-2 font-display text-[clamp(1.5rem,1.1rem+1.6vw,2.4rem)] leading-tight tracking-tight text-text-primary">
          Banks share counts. The tree learns the rest.
        </h2>
        <p className="mt-4 text-[13px] leading-relaxed text-text-secondary">
          Histogram telemetry is unavailable right now. The federated GBDT still chooses its splits
          from securely-summed bank histograms — this visualization will resume once the data feed
          returns.
        </p>
      </section>
    );
  }

  return (
    <section
      aria-labelledby="gbdt-heading"
      className="rise overflow-hidden rounded-[20px] border bg-bg-surface/80 backdrop-blur-sm"
      style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-panel)" }}
    >
      <header
        className="grid gap-4 border-b p-5 sm:p-6 lg:grid-cols-[1fr_auto] lg:items-end"
        style={{ borderColor: "var(--border-default)" }}
      >
        <div>
          <p className="eyebrow text-accent-gold">Federated GBDT · histograms, not data</p>
          <h2
            id="gbdt-heading"
            className="mt-2 font-display text-[clamp(1.7rem,1.2rem+2vw,3rem)] leading-tight tracking-tight text-text-primary"
          >
            Banks share counts. The tree learns the rest.
          </h2>
          <p className="mt-3 max-w-3xl text-[13px] leading-relaxed text-text-secondary sm:text-[14px]">
            Each of {data?.bankHistograms.length ?? 4} banks contributes only a privacy-masked
            histogram of the “{data?.feature ?? "amount"}” feature — bin counts, never customer rows.
            The coordinator securely sums them into one global histogram and picks the best-gain
            split from that aggregate alone.
          </p>
        </div>
        <div
          className="grid grid-cols-2 gap-px overflow-hidden rounded-[14px] border text-center sm:min-w-[300px]"
          style={{ borderColor: "var(--border-default)", background: "var(--border-default)" }}
        >
          <Stat
            label="banks summed"
            value={`${banksIn}/${data?.bankHistograms.length ?? 0}`}
            tone="fed"
          />
          <Stat
            label="split gain"
            value={data ? data.split.gain.toFixed(0) : "—"}
            tone="gold"
          />
        </div>
      </header>

      <div className="p-4 sm:p-5">
        <div
          ref={wrapRef}
          className="relative h-[440px] w-full overflow-hidden rounded-[14px] border bg-bg-deep sm:h-[520px]"
          style={{ borderColor: "var(--border-default)" }}
        >
          <canvas
            ref={canvasRef}
            className="block h-full w-full"
            role="img"
            aria-label="Per-bank feature histograms stacking into a securely-summed global histogram, then a split and a small decision tree"
          />

          {/* Privacy note overlay */}
          <div
            className="pointer-events-none absolute left-3 top-3 max-w-[260px] rounded-[10px] border px-3 py-2 text-[11px] leading-snug backdrop-blur-sm"
            style={{
              borderColor: "var(--border-default)",
              background: "rgba(11,16,28,0.7)",
              color: "var(--text-secondary)",
            }}
          >
            Per-bank histograms are pairwise-masked and secure-summed — no bank ever sees another’s
            counts, and no raw rows are shared.
          </div>

          {/* Caption overlay */}
          <p
            className="pointer-events-none absolute bottom-3 left-3 right-3 text-[12px] leading-snug text-text-secondary sm:text-[13px]"
            style={{ textShadow: "0 1px 8px rgba(8,12,20,0.92)" }}
          >
            {caption}
          </p>
        </div>

        <p className="tabular mt-3 text-[12px] text-text-muted">
          {data?.meta.model ?? "federated histogram GBDT"}
          {typeof data?.meta.metric === "number"
            ? ` · recall ${(data.meta.metric * 100).toFixed(1)}%`
            : ""}
          {" · global histogram = secure sum of every bank’s counts"}
        </p>
      </div>
    </section>
  );
}

// ── Tree renderer (pure canvas; no per-frame allocation beyond locals) ───────
function drawTree(
  ctx: CanvasRenderingContext2D,
  tree: TreeNode[],
  x: number,
  y: number,
  w: number,
  h: number,
  phase: number,
) {
  // Assign depth by walking parents; lay out by depth band, centre per row.
  const n = tree.length;
  if (!n) return;
  const depth = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    const p = tree[i].parent;
    depth[i] = p === null || p < 0 ? 0 : depth[p] + 1;
  }
  let maxDepth = 0;
  for (let i = 0; i < n; i++) if (depth[i] > maxDepth) maxDepth = depth[i];

  // group node ids by depth
  const byDepth: number[][] = [];
  for (let d = 0; d <= maxDepth; d++) byDepth.push([]);
  for (let i = 0; i < n; i++) byDepth[depth[i]].push(i);

  const rowH = maxDepth > 0 ? h / (maxDepth + 1) : h;
  const cx = new Array<number>(n).fill(x + w / 2);
  const cy = new Array<number>(n).fill(y);
  for (let d = 0; d <= maxDepth; d++) {
    const row = byDepth[d];
    const slot = w / (row.length + 1);
    for (let k = 0; k < row.length; k++) {
      cx[row[k]] = x + slot * (k + 1);
      cy[row[k]] = y + rowH * d + rowH * 0.5;
    }
  }

  // edges first (parent -> child), revealed with phase
  ctx.lineWidth = 1.4;
  for (let i = 0; i < n; i++) {
    const p = tree[i].parent;
    if (p === null || p < 0) continue;
    const ep = clamp01((phase - 0.15) * 1.4);
    if (ep <= 0) continue;
    const x1 = cx[p];
    const y1 = cy[p];
    const x2 = cx[i];
    const y2 = cy[i];
    ctx.strokeStyle = "rgba(150,170,200,0.35)";
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 + (x2 - x1) * ep, y1 + (y2 - y1) * ep);
    ctx.stroke();
    // L / R label at midpoint
    if (ep > 0.9 && tree[i].side) {
      ctx.fillStyle = "rgba(176,192,214,0.6)";
      ctx.font = "600 9px ui-sans-serif, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(tree[i].side as string, (x1 + x2) / 2, (y1 + y2) / 2);
    }
  }

  // nodes
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let i = 0; i < n; i++) {
    const np = clamp01((phase - depth[i] * 0.12) * 1.6);
    if (np <= 0) continue;
    const isLeaf = tree[i].feature === "leaf";
    const r = (isLeaf ? 5 : 7) * np;
    ctx.beginPath();
    ctx.arc(cx[i], cy[i], r, 0, Math.PI * 2);
    if (isLeaf) {
      ctx.fillStyle = `rgba(120,150,170,${(0.7 * np).toFixed(3)})`;
      ctx.fill();
    } else {
      ctx.fillStyle = `hsla(42,90%,58%,${(0.92 * np).toFixed(3)})`;
      ctx.fill();
      // label: feature ≤ threshold
      if (np > 0.85) {
        ctx.fillStyle = "rgba(246,233,205,0.92)";
        ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
        ctx.fillText(
          `${tree[i].feature} ≤ ${tree[i].threshold.toFixed(1)}`,
          cx[i],
          cy[i] - 14,
        );
      }
    }
  }
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

const FederatedGbdt = memo(FederatedGbdtInner);
export default FederatedGbdt;
