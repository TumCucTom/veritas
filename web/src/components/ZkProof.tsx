"use client";
import { memo, useEffect, useMemo, useRef, useState } from "react";

/*
  ZkProof — visualizes Veritas's VSA verifiable-norm proof.

  The network proves an aggregated update is well-behaved (its L2 norm is within
  a public bound) WITHOUT ever seeing the update: a Pedersen commitment hides the
  value, and a Fiat-Shamir sigma range proof certifies "||u|| <= B" against the
  commitment only. The server verifies and REJECTS norm-violating poison before
  aggregation — all without revealing the raw update.

  Data: /viz/vsa.json (a REAL protocol run captured offline by tools/gen_vsa_viz.py).
  Auto-cycles honest (accepted) -> poison (rejected), looping. The proof steps
  (commit -> challenge -> response -> verify) advance as a stepper.

  Self-contained client component, no props. Single RAF in refs, IntersectionObserver
  pause offscreen, dpr cap 2, no per-frame allocation, graceful fetch fallback,
  prefers-reduced-motion holds a static frame.
*/

type VsaClient = {
  id: string;
  norm: number;
  committed: boolean;
  commitmentPreview: string;
  accepted: boolean;
  poison: boolean;
};

type VsaData = {
  bound: number;
  clients: VsaClient[];
  steps: string[];
  meta: { note: string; scheme: string };
};

const STEP_LABELS: Record<string, { label: string; caption: string }> = {
  commit: {
    label: "Commit",
    caption: "Client publishes a Pedersen commitment to ‖u‖² — the value is sealed, never revealed.",
  },
  challenge: {
    label: "Challenge",
    caption: "A Fiat-Shamir challenge is derived from the transcript — non-interactive, unforgeable.",
  },
  response: {
    label: "Response",
    caption: "Client answers with a sigma-protocol response binding the bits of the hidden norm.",
  },
  verify: {
    label: "Verify",
    caption: "Verifier checks the proof against the commitment only — learns one bit: in-bounds or not.",
  },
};

const STEP_MS = 1400; // dwell per protocol step
const VERDICT_HOLD_MS = 2200; // extra dwell on the final "verify" step
const TWEEN_MS = 600;

// Theme tokens (mirror globals.css). RGB triples for cheap alpha variation.
const TEAL = "52, 211, 153";
const ROSE = "240, 74, 82";
const GOLD = "214, 168, 91";

function clamp01(t: number): number {
  return t < 0 ? 0 : t > 1 ? 1 : t;
}
function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

function ZkProof() {
  const [data, setData] = useState<VsaData | null>(null);
  const [error, setError] = useState(false);
  const [step, setStep] = useState(0); // active protocol step 0..steps-1
  const [clientIdx, setClientIdx] = useState(0); // which client is on stage
  const [paused, setPaused] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Refs drive the RAF loop so it is never recreated per render.
  const stepRef = useRef(0);
  const clientRef = useRef(0);
  const tweenStartRef = useRef(0);
  const visibleRef = useRef(true);
  const hoverPausedRef = useRef(false);
  const reduceMotionRef = useRef(false);
  const dataRef = useRef<VsaData | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/viz/vsa.json")
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status));
        return r.json();
      })
      .then((json: VsaData) => {
        if (cancelled) return;
        if (!json?.clients?.length || !json?.steps?.length)
          throw new Error("empty");
        dataRef.current = json;
        setData(json);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Mirror state into refs + arm the tween clock on any stage change.
  useEffect(() => {
    if (step !== stepRef.current || clientIdx !== clientRef.current) {
      stepRef.current = step;
      clientRef.current = clientIdx;
      tweenStartRef.current = performance.now();
    }
  }, [step, clientIdx]);

  // Auto-advance steps; loop honest -> poison after the last step.
  useEffect(() => {
    if (!data || paused) return;
    const nSteps = data.steps.length;
    const isLast = step === nSteps - 1;
    const id = window.setTimeout(
      () => {
        if (isLast) {
          setClientIdx((c) => (c + 1) % data.clients.length);
          setStep(0);
        } else {
          setStep((s) => s + 1);
        }
      },
      isLast ? VERDICT_HOLD_MS : STEP_MS,
    );
    return () => window.clearTimeout(id);
  }, [data, step, clientIdx, paused]);

  // Pause when offscreen.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        const vis = entries[0]?.isIntersecting ?? true;
        visibleRef.current = vis;
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

  // The RAF render loop — created once, reads everything from refs.
  useEffect(() => {
    if (!data) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    reduceMotionRef.current = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    let width = 0;
    let height = 0;
    const sizeCanvas = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      width = Math.max(320, Math.floor(rect.width));
      height = Math.max(220, Math.floor(rect.height));
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    sizeCanvas();

    let raf = 0;
    const render = (now: number) => {
      const d = dataRef.current;
      if (d) {
        const reduce = reduceMotionRef.current;
        const t = reduce
          ? 1
          : easeInOut(clamp01((now - tweenStartRef.current) / TWEEN_MS));
        drawScene(
          ctx,
          d,
          clientRef.current,
          stepRef.current,
          t,
          now,
          width,
          height,
          reduce,
        );
      }
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

  const client = data?.clients[clientIdx] ?? null;
  const stepKey = data?.steps[step] ?? "commit";
  const meta = data?.meta;

  const verdict = useMemo(() => {
    if (!client) return null;
    return client.accepted
      ? { ok: true, label: "in-bounds · accepted" }
      : { ok: false, label: "norm exceeds bound · rejected" };
  }, [client]);

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
    hoverPausedRef.current = true;
    setPaused(true);
    setStep((s) => (s - 1 + data.steps.length) % data.steps.length);
  };
  const goNext = () => {
    if (!data) return;
    hoverPausedRef.current = true;
    setPaused(true);
    setStep((s) => (s + 1) % data.steps.length);
  };

  return (
    <section
      ref={containerRef}
      aria-labelledby="zk-proof-heading"
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
            Verifiable norm proof · zero-knowledge
          </p>
          <h2
            id="zk-proof-heading"
            className="mt-2 font-display text-[clamp(1.7rem,1.2rem+2vw,3rem)] leading-tight tracking-tight text-text-primary"
          >
            Prove it&rsquo;s clean — without revealing it.
          </h2>
          <p className="mt-3 max-w-2xl text-[13px] leading-relaxed text-text-secondary sm:text-[14px]">
            Each client commits to its update&rsquo;s squared norm and proves —
            in zero knowledge — that it stays within the bound. The verifier
            checks the proof against the commitment only, rejecting poison{" "}
            <em>without ever seeing the update</em>.
          </p>
        </div>
        <div
          className="grid grid-cols-2 gap-px overflow-hidden rounded-[14px] border text-center sm:min-w-[260px]"
          style={{
            borderColor: "var(--border-default)",
            background: "var(--border-default)",
          }}
        >
          <Metric label="Norm bound B" value={meta ? data!.bound.toFixed(1) : "—"} tone="gold" />
          <Metric label="Scheme" value="Pedersen σ" tone="text" />
        </div>
      </header>

      {error ? (
        <div className="p-4 sm:p-5">
          <Fallback />
        </div>
      ) : (
        <>
          {/* Stepper */}
          <div
            className="flex flex-wrap items-center gap-2 border-b px-5 py-3 sm:px-6"
            style={{ borderColor: "var(--border-default)" }}
          >
            {(data?.steps ?? ["commit", "challenge", "response", "verify"]).map(
              (s, i) => {
                const active = i === step;
                const done = i < step;
                const color = active
                  ? "var(--accent-gold)"
                  : done
                    ? "var(--fed)"
                    : "var(--text-muted)";
                const meta2 = STEP_LABELS[s] ?? { label: s, caption: "" };
                return (
                  <button
                    key={s}
                    type="button"
                    onClick={() => {
                      if (!data) return;
                      hoverPausedRef.current = true;
                      setPaused(true);
                      setStep(i);
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
                    {meta2.label}
                  </button>
                );
              },
            )}
          </div>

          <div className="p-4 sm:p-5">
            {/* Prover -> Verifier stage */}
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)_minmax(0,1fr)]">
              {/* LEFT: prover — committed, value hidden */}
              <Panel title="Prover" subtitle={client ? client.id : ""}>
                <div className="relative grid place-items-center py-3">
                  <div
                    aria-hidden
                    className="grid h-24 w-24 place-items-center rounded-2xl border"
                    style={{
                      borderColor: "var(--border-strong)",
                      background:
                        "repeating-linear-gradient(45deg, rgba(111,122,144,0.10) 0 6px, rgba(111,122,144,0.02) 6px 12px)",
                      filter: "blur(0.4px)",
                    }}
                  >
                    {/* blurred / masked blob */}
                    <span
                      className="h-12 w-12 rounded-full"
                      style={{
                        background:
                          "radial-gradient(circle, rgba(111,122,144,0.55), rgba(111,122,144,0.05))",
                        filter: "blur(6px)",
                      }}
                    />
                    {/* lock glyph */}
                    <span
                      className="absolute text-[20px]"
                      style={{ color: "var(--text-muted)" }}
                      aria-hidden
                    >
                      🔒
                    </span>
                  </div>
                </div>
                <p className="text-center text-[11px] uppercase tracking-[0.14em] text-text-muted">
                  committed — value hidden
                </p>
                <p
                  className="tabular mt-2 truncate text-center text-[11px]"
                  style={{ color: "var(--text-secondary)" }}
                  title={client?.commitmentPreview}
                >
                  C: {client ? client.commitmentPreview : "…"}
                </p>
              </Panel>

              {/* MIDDLE: the proof channel */}
              <Panel
                title="Range proof"
                subtitle={STEP_LABELS[stepKey]?.label ?? stepKey}
              >
                <canvas
                  ref={canvasRef}
                  className="block h-[150px] w-full rounded-[10px] border bg-bg-deep"
                  style={{ borderColor: "var(--border-default)" }}
                  role="img"
                  aria-label={`Zero-knowledge range proof, ${
                    STEP_LABELS[stepKey]?.label ?? stepKey
                  } step, client ${client?.id ?? ""}`}
                />
                <p
                  key={`${clientIdx}-${step}`}
                  className="rise mt-3 min-h-[2.5rem] text-[12px] leading-relaxed text-text-secondary"
                >
                  {STEP_LABELS[stepKey]?.caption ?? ""}
                </p>
              </Panel>

              {/* RIGHT: verifier verdict + gauge */}
              <Panel title="Verifier" subtitle="verdict">
                {client && verdict && (
                  <>
                    <div
                      key={`verdict-${clientIdx}`}
                      className="rise grid place-items-center py-2"
                    >
                      <span
                        className="grid h-16 w-16 place-items-center rounded-full text-[30px] font-bold"
                        style={{
                          color: verdict.ok ? "var(--fed)" : "var(--silo)",
                          background: verdict.ok
                            ? "rgba(52,211,153,0.10)"
                            : "rgba(240,74,82,0.10)",
                          border: `1px solid ${
                            verdict.ok
                              ? "rgba(52,211,153,0.45)"
                              : "rgba(240,74,82,0.45)"
                          }`,
                          // verdict only crystallizes on the final verify step
                          opacity: step === (data?.steps.length ?? 4) - 1 ? 1 : 0.32,
                          transition: "opacity 400ms ease",
                        }}
                        aria-hidden
                      >
                        {verdict.ok ? "✓" : "✗"}
                      </span>
                    </div>
                    <p
                      className="text-center text-[12px] font-semibold"
                      style={{
                        color: verdict.ok ? "var(--fed)" : "var(--silo)",
                      }}
                    >
                      {verdict.label}
                    </p>
                    <Gauge norm={client.norm} bound={data!.bound} ok={verdict.ok} />
                  </>
                )}
              </Panel>
            </div>

            {/* Footer: takeaway + transport */}
            <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <p className="max-w-2xl text-[12px] leading-relaxed text-text-muted">
                {meta?.note}
              </p>
              <div className="flex shrink-0 items-center gap-2">
                <TransportButton label="Prev step" onClick={goPrev}>
                  ‹ Prev
                </TransportButton>
                <TransportButton label="Next step" onClick={goNext}>
                  Next ›
                </TransportButton>
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  );
}

// --------------------------------------------------------------------------- //
// Canvas: the proof channel. Particles flow prover -> verifier per step; on the
// final "verify" step the channel resolves to green (accept) or red (reject).
// --------------------------------------------------------------------------- //
function drawScene(
  ctx: CanvasRenderingContext2D,
  data: VsaData,
  clientIdx: number,
  step: number,
  t: number,
  now: number,
  width: number,
  height: number,
  reduce: boolean,
) {
  const client = data.clients[clientIdx];
  const nSteps = data.steps.length;
  const isVerify = step === nSteps - 1;
  const accepted = client.accepted;

  ctx.clearRect(0, 0, width, height);
  const bg = ctx.createLinearGradient(0, 0, width, height);
  bg.addColorStop(0, "rgba(10,14,22,0.98)");
  bg.addColorStop(1, "rgba(13,19,32,0.98)");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);

  const midY = height / 2;
  const xL = 26;
  const xR = width - 26;

  // Channel line.
  const channelRGB = isVerify ? (accepted ? TEAL : ROSE) : GOLD;
  ctx.save();
  ctx.strokeStyle = `rgba(${channelRGB}, 0.35)`;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(xL, midY);
  ctx.lineTo(xR, midY);
  ctx.stroke();
  ctx.restore();

  // Prover node (left) and verifier node (right).
  drawNode(ctx, xL, midY, "111,122,144", "P");
  drawNode(
    ctx,
    xR,
    midY,
    isVerify ? (accepted ? TEAL : ROSE) : "111,122,144",
    "V",
  );

  // Flowing proof particles along the channel. Progress depends on the step so
  // commit/challenge/response/verify visibly advance the transcript.
  const baseProgress = (step + (reduce ? 1 : t)) / nSteps;
  const flow = reduce ? 0 : (now / 1400) % 1;
  const COUNT = 7;
  for (let i = 0; i < COUNT; i++) {
    // staggered particles within the portion of the channel "filled" so far
    let p = (flow + i / COUNT) % 1;
    p *= baseProgress; // only travel up to the current progress
    const x = xL + (xR - xL) * p;
    const wobble = reduce ? 0 : Math.sin(now / 300 + i) * 3;
    const a = 0.25 + 0.5 * (1 - Math.abs(0.5 - p) * 2);
    ctx.fillStyle = `rgba(${channelRGB}, ${a})`;
    ctx.beginPath();
    ctx.arc(x, midY + wobble, 2.6, 0, Math.PI * 2);
    ctx.fill();
  }

  // On verify, pulse a verdict halo at the verifier.
  if (isVerify) {
    const pulse = reduce ? 0.6 : 0.5 + 0.5 * Math.sin(now / 400);
    const haloR = 16 + 6 * pulse;
    const g = ctx.createRadialGradient(xR, midY, 0, xR, midY, haloR);
    g.addColorStop(0, `rgba(${channelRGB}, ${0.45 * (reduce ? 1 : t)})`);
    g.addColorStop(1, `rgba(${channelRGB}, 0)`);
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(xR, midY, haloR, 0, Math.PI * 2);
    ctx.fill();
  }

  // Step label under the channel.
  ctx.fillStyle = `rgba(${channelRGB}, 0.9)`;
  ctx.font = "600 10px ui-monospace, 'SF Mono', monospace";
  ctx.textAlign = "center";
  const label = (STEP_LABELS[data.steps[step]]?.label ?? data.steps[step]).toUpperCase();
  ctx.fillText(label, width / 2, midY - 14);
}

function drawNode(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  rgb: string,
  glyph: string,
) {
  ctx.save();
  const glow = ctx.createRadialGradient(x, y, 0, x, y, 18);
  glow.addColorStop(0, `rgba(${rgb}, 0.5)`);
  glow.addColorStop(1, `rgba(${rgb}, 0)`);
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(x, y, 18, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = `rgba(${rgb}, 0.95)`;
  ctx.beginPath();
  ctx.arc(x, y, 9, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "rgba(242,239,230,0.95)";
  ctx.font = "700 10px ui-monospace, 'SF Mono', monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(glyph, x, y);
  ctx.textBaseline = "alphabetic";
  ctx.restore();
}

// --------------------------------------------------------------------------- //
// Subcomponents
// --------------------------------------------------------------------------- //
function Panel({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className="rounded-[14px] border bg-bg-surface-2 p-4"
      style={{ borderColor: "var(--border-default)" }}
    >
      <div className="mb-1 flex items-baseline justify-between">
        <p className="eyebrow text-text-muted">{title}</p>
        {subtitle && (
          <p className="tabular text-[11px] uppercase tracking-[0.14em] text-accent-gold">
            {subtitle}
          </p>
        )}
      </div>
      {children}
    </div>
  );
}

function Gauge({
  norm,
  bound,
  ok,
}: {
  norm: number;
  bound: number;
  ok: boolean;
}) {
  // Scale so the bound sits at ~62% of the track; over-bound norms run past it.
  const boundFrac = 0.62;
  const max = bound / boundFrac;
  const fill = Math.min(1, norm / max);
  const color = ok ? "var(--fed)" : "var(--silo)";
  return (
    <div className="mt-3">
      <div
        className="relative h-2.5 w-full overflow-hidden rounded-full"
        style={{ background: "var(--bg-deep)" }}
      >
        <div
          className="h-full rounded-full"
          style={{
            width: `${fill * 100}%`,
            background: color,
            transition: "width 500ms ease",
          }}
        />
        {/* bound marker */}
        <span
          aria-hidden
          className="absolute top-[-3px] h-[14px] w-[2px]"
          style={{
            left: `${boundFrac * 100}%`,
            background: "var(--accent-gold)",
          }}
        />
      </div>
      <div className="mt-1.5 flex justify-between text-[10px] uppercase tracking-[0.12em] text-text-muted">
        <span className="tabular" style={{ color }}>
          ‖u‖ {norm.toFixed(2)}
        </span>
        <span className="tabular" style={{ color: "var(--accent-gold)" }}>
          B {bound.toFixed(1)}
        </span>
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "text" | "gold";
}) {
  const color = tone === "gold" ? "var(--accent-gold)" : "var(--text-primary)";
  return (
    <div className="bg-bg-surface-2 px-4 py-3">
      <p className="eyebrow text-text-muted">{label}</p>
      <p
        className="tabular mt-1 font-display text-xl leading-none"
        style={{ color }}
      >
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

function Fallback() {
  return (
    <div
      className="grid h-[260px] w-full place-items-center rounded-[14px] border bg-bg-deep text-center"
      style={{ borderColor: "var(--border-default)" }}
    >
      <div className="max-w-sm px-6">
        <p className="eyebrow" style={{ color: "var(--silo)" }}>
          Visualization unavailable
        </p>
        <p className="mt-2 text-[13px] leading-relaxed text-text-secondary">
          Each client commits to its update&rsquo;s squared norm and proves, in
          zero knowledge, that it stays within the bound. The verifier checks the
          proof against the commitment alone — rejecting norm-violating poison
          without ever seeing the raw update.
        </p>
      </div>
    </div>
  );
}

export default memo(ZkProof);
