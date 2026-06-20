"use client";
import { useControlPlane } from "../lib/controlPlaneStore";
import type { EdgeFleet } from "../lib/controlplane-types";
import { formatNumber } from "../lib/format";

/**
 * Edge fleet — Tier-0 coverage (spec §7.1 last bullet): app versions, % of
 * customers on the latest model, scam-in-progress alert volumes.
 *
 * The production control plane may not yet expose edge-fleet metrics, so we
 * render live numbers ONLY when tenant state carries them. Otherwise we show a
 * clearly-labelled REPRESENTATIVE panel — never fabricated live numbers without
 * the label.
 */
const REPRESENTATIVE: Required<EdgeFleet> = {
  devicesEnrolled: 1_840_000,
  onLatestModelPct: 0.82,
  modelVersion: 0,
  scamInProgressAlerts24h: 312,
  appVersions: [
    { version: "5.4.x", share: 0.61 },
    { version: "5.3.x", share: 0.27 },
    { version: "≤5.2", share: 0.12 },
  ],
};

export default function EdgeFleetPanel() {
  const { state, status } = useControlPlane();
  const live = state?.edgeFleet;
  const fleet: EdgeFleet = live ?? REPRESENTATIVE;
  const isLive = Boolean(live);

  const onLatest = Math.round((fleet.onLatestModelPct ?? 0) * 100);
  const modelVersion = isLive ? fleet.modelVersion : state?.modelVersion;

  return (
    <section
      aria-labelledby="edge-heading"
      className="flex h-full flex-col rounded-[20px] border bg-bg-surface/70 p-5 backdrop-blur-sm sm:p-6"
      style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-raise)" }}
    >
      <header className="mb-4 flex items-start justify-between gap-3">
        <div>
          <p className="eyebrow text-accent-gold">Tier 0 · on-device</p>
          <h2 id="edge-heading" className="font-display text-xl tracking-tight text-text-primary">
            Edge fleet coverage
          </h2>
        </div>
        <SourceBadge isLive={isLive} />
      </header>

      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-[14px] border"
        style={{ borderColor: "var(--border-default)", background: "var(--border-default)" }}>
        <Metric
          label="Devices enrolled"
          value={fleet.devicesEnrolled != null ? formatNumber(fleet.devicesEnrolled) : "—"}
          sub="customer devices running the SDK"
        />
        <Metric
          label="On latest model"
          value={`${onLatest}%`}
          sub={`edge model v${modelVersion ?? "—"}`}
          accent="var(--fed)"
        />
        <Metric
          label="Scam-in-progress · 24h"
          value={fleet.scamInProgressAlerts24h != null ? formatNumber(fleet.scamInProgressAlerts24h) : "—"}
          sub="victim-side alerts raised on-device"
          accent="var(--accent-gold)"
        />
        <Metric
          label="Data leaving device"
          value="0"
          sub="DP-protected updates only"
          accent="var(--fed)"
        />
      </div>

      {/* App version distribution */}
      <div className="mt-4">
        <p className="eyebrow mb-2 text-text-muted">App version distribution</p>
        <div className="flex flex-col gap-1.5">
          {(fleet.appVersions ?? []).map((v) => (
            <div key={v.version} className="flex items-center gap-2.5">
              <span className="tabular w-14 shrink-0 text-[11px] text-text-secondary">{v.version}</span>
              <div className="h-2 flex-1 overflow-hidden rounded-full" style={{ background: "var(--bg-deep)" }}>
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${Math.round(v.share * 100)}%`,
                    background: "linear-gradient(90deg, var(--accent-gold-soft), var(--accent-gold))",
                  }}
                />
              </div>
              <span className="tabular w-9 shrink-0 text-right text-[11px] text-text-muted">
                {Math.round(v.share * 100)}%
              </span>
            </div>
          ))}
        </div>
      </div>

      <p className="mt-auto pt-5 text-[11px] leading-snug text-text-muted">
        {isLive
          ? "Live Tier-0 coverage from this tenant’s node."
          : status === "offline"
            ? "Control plane offline · representative coverage shown — not live figures."
            : "Representative coverage · the control plane does not yet expose Tier-0 fleet metrics for this tenant."}
      </p>
    </section>
  );
}

function Metric({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub: string;
  accent?: string;
}) {
  return (
    <div className="bg-bg-surface p-4">
      <p className="eyebrow text-text-muted">{label}</p>
      <p className="tabular mt-2 font-display text-2xl leading-none tracking-tight"
        style={{ color: accent ?? "var(--text-primary)" }}>
        {value}
      </p>
      <p className="mt-1.5 text-[11px] text-text-muted">{sub}</p>
    </div>
  );
}

function SourceBadge({ isLive }: { isLive: boolean }) {
  return (
    <span
      className="shrink-0 inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em]"
      style={{
        background: "var(--bg-surface-2)",
        color: isLive ? "var(--fed)" : "var(--accent-gold)",
      }}
    >
      <span aria-hidden className="h-1.5 w-1.5 rounded-full"
        style={{ background: isLive ? "var(--fed)" : "var(--accent-gold)" }} />
      {isLive ? "live" : "representative"}
    </span>
  );
}
