"use client";
import { useControlPlane } from "../lib/controlPlaneStore";
import { controlPlane } from "../lib/controlplane";
import type { ModelRegistryEntry, ModelStatus } from "../lib/controlplane-types";

/**
 * Governance / network view (spec §7.1 "Network health" + §7.2 model
 * governance). One tenant sees only their own node + the federation aggregate:
 *   - tenant / member identity and the customerRecordsTransmitted: 0 proof,
 *   - node attestation + sync status,
 *   - the global model registry with version / status / metrics.
 *
 * All from the control plane; degrades to an offline notice.
 */
export default function GovernancePanel() {
  const { status, tenantId, state, registry } = useControlPlane();
  const node = state?.node;
  const records = state?.customerRecordsTransmitted ?? 0;
  const recordsClean = records === 0;
  const attested = node?.attestation === "verified";

  return (
    <section
      aria-labelledby="governance-heading"
      className="flex h-full flex-col rounded-[20px] border bg-bg-surface/70 p-5 backdrop-blur-sm sm:p-6"
      style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-raise)" }}
    >
      <header className="mb-4 flex items-start justify-between gap-3">
        <div>
          <p className="eyebrow text-accent-gold">Network health &amp; governance</p>
          <h2 id="governance-heading" className="font-display text-xl tracking-tight text-text-primary">
            This tenant&apos;s node &amp; the federation
          </h2>
        </div>
        <PlaneBadge status={status} />
      </header>

      {/* Identity + sovereignty proof row */}
      <div className="grid grid-cols-1 gap-px overflow-hidden rounded-[14px] border sm:grid-cols-3"
        style={{ borderColor: "var(--border-default)", background: "var(--border-default)" }}>
        <Cell label="Tenant / member">
          <span className="font-mono text-[13px] text-text-primary">{tenantId}</span>
          <span className="mt-1 block text-[11px] text-text-muted">
            model v{state?.modelVersion ?? "—"} · round {state?.round ?? "—"}
          </span>
        </Cell>
        <Cell label="Records transmitted">
          <span
            className="tabular font-display text-2xl leading-none"
            style={{ color: recordsClean ? "var(--fed)" : "var(--silo)" }}
          >
            {records}
          </span>
          <span className="mt-1 block text-[11px] text-text-muted">
            {recordsClean ? "sovereignty proof — only model deltas leave" : "egress detected"}
          </span>
        </Cell>
        <Cell label="Node attestation">
          <span className="flex items-center gap-2">
            <span aria-hidden className="h-2 w-2 rounded-full"
              style={{ background: attested ? "var(--fed)" : "var(--text-muted)" }} />
            <span className="text-[13px] font-semibold"
              style={{ color: attested ? "var(--fed)" : "var(--text-secondary)" }}>
              {node?.attestation ?? (status === "offline" ? "unavailable" : "unknown")}
            </span>
          </span>
          <span className="mt-1 block text-[11px] text-text-muted">
            {node?.status ? `${node.status} · synced ${formatSync(node.lastSync)}` : "awaiting node telemetry"}
          </span>
        </Cell>
      </div>

      {/* Model registry */}
      <div className="mt-5">
        <p className="eyebrow mb-2 text-text-muted">Global model registry</p>
        {registry.length === 0 ? (
          <p className="rounded-[12px] border border-dashed p-3 text-[12px] text-text-muted"
            style={{ borderColor: "var(--border-default)" }}>
            {status === "offline"
              ? "Control plane offline — model registry unavailable."
              : "No model versions registered yet."}
          </p>
        ) : (
          <ol className="flex flex-col gap-1.5">
            {registry.map((m) => (
              <RegistryRow key={m.version} entry={m} />
            ))}
          </ol>
        )}
      </div>

      <p className="mt-auto pt-5 text-[11px] leading-snug text-text-muted">
        SR 11-7-grade governance: every promotion / rollback is attested in the transparency log.
      </p>
    </section>
  );
}

function RegistryRow({ entry }: { entry: ModelRegistryEntry }) {
  const recall = entry.metrics?.recall;
  return (
    <li
      className="flex items-center justify-between gap-3 rounded-[12px] border bg-bg-deep/50 px-3 py-2"
      style={{ borderColor: "var(--border-default)" }}
    >
      <div className="flex items-center gap-2.5">
        <span className="tabular rounded-[7px] px-2 py-0.5 text-[11px] font-semibold text-bg-deep"
          style={{ background: "var(--accent-gold)" }}>
          v{entry.version}
        </span>
        <StatusPill status={entry.status} />
      </div>
      <div className="flex items-baseline gap-1.5">
        <span className="eyebrow">recall</span>
        <span className="tabular text-[14px] font-semibold"
          style={{ color: typeof recall === "number" && recall >= 0.75 ? "var(--fed)" : "var(--text-secondary)" }}>
          {typeof recall === "number" ? `${Math.round(recall * 100)}%` : "—"}
        </span>
      </div>
    </li>
  );
}

const STATUS_STYLE: Record<ModelStatus, { bg: string; fg: string }> = {
  promoted: { bg: "rgba(52,211,153,0.16)", fg: "var(--fed)" },
  canary: { bg: "rgba(214,168,91,0.16)", fg: "var(--accent-gold)" },
  rolledback: { bg: "rgba(240,74,82,0.16)", fg: "var(--silo)" },
};

function StatusPill({ status }: { status: ModelStatus }) {
  const s = STATUS_STYLE[status] ?? { bg: "var(--bg-surface-2)", fg: "var(--text-muted)" };
  return (
    <span
      className="rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.1em]"
      style={{ background: s.bg, color: s.fg }}
    >
      {status}
    </span>
  );
}

function Cell({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="bg-bg-surface p-3.5">
      <p className="eyebrow text-text-muted">{label}</p>
      <div className="mt-1.5">{children}</div>
    </div>
  );
}

function PlaneBadge({ status }: { status: "connecting" | "live" | "offline" }) {
  const live = status === "live";
  const offline = status === "offline";
  const color = live ? "var(--fed)" : offline ? "var(--silo)" : "var(--text-muted)";
  return (
    <span
      className="shrink-0 inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em]"
      style={{ background: "var(--bg-surface-2)", color }}
    >
      <span aria-hidden className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
      {live ? "control plane · live" : offline ? "control plane · offline" : "connecting…"}
    </span>
  );
}

function formatSync(ts?: string): string {
  if (!ts) return "—";
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}
