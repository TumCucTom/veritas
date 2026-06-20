"use client";
import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import { useEvents } from "../lib/useEvents";
import { useControlPlane } from "../lib/controlPlaneStore";
import { controlPlane } from "../lib/controlplane";
import type { TransparencyRoot } from "../lib/controlplane-types";
import type { Provenance, VeritasEvent } from "../lib/types";

/**
 * Provenance ledger — the model bill-of-materials.
 *
 * Primary source is the control plane (GET /v1/tenants/{tid}/provenance). If
 * the control plane is offline we fall back to the node's /provenance so the
 * single-node demo keeps working. The "verify" affordance resolves the signed
 * Merkle root (GET /v1/transparency/root) — tamper-evidence, on demand.
 */
export default function ProvenancePanel() {
  const cp = useControlPlane();
  const [nodeRecords, setNodeRecords] = useState<Provenance[]>([]);

  // Node fallback — only consulted when the control plane has nothing.
  const loadNode = useCallback(async (): Promise<void> => {
    try {
      const next = await api.provenance();
      setNodeRecords(Array.isArray(next) ? next : []);
    } catch {
      // Node provenance unreachable — keep the last known ledger.
    }
  }, []);

  useEffect(() => {
    void loadNode();
  }, [loadNode]);

  const onEvent = useCallback(
    (e: VeritasEvent): void => {
      if (e.type === "round_complete") void loadNode();
    },
    [loadNode],
  );
  useEvents(onEvent);

  const rows: Provenance[] = cp.provenance.length > 0 ? cp.provenance : nodeRecords;
  const source: "control-plane" | "node" =
    cp.provenance.length > 0 ? "control-plane" : "node";

  return (
    <section
      aria-labelledby="provenance-heading"
      className="flex h-full flex-col rounded-[20px] border bg-bg-surface/70 p-5 backdrop-blur-sm sm:p-6"
      style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-raise)" }}
    >
      <header className="mb-4 flex items-start justify-between gap-3">
        <div>
          <p className="eyebrow text-accent-gold">Model bill-of-materials</p>
          <h2
            id="provenance-heading"
            className="font-display text-xl tracking-tight text-text-primary"
          >
            Provenance ledger
          </h2>
        </div>
        <VerifyControl />
      </header>

      {rows.length === 0 ? (
        <EmptyState offline={cp.status === "offline"} />
      ) : (
        <div className="flex flex-col gap-2.5">
          <Legend source={source} />
          <ol className="flex flex-col gap-2">
            {rows.map((rec) => (
              <ProvenanceRow key={rec.round} record={rec} />
            ))}
          </ol>
        </div>
      )}

      <p className="mt-auto pt-5 text-[11px] leading-snug text-text-muted">
        Signed Merkle transparency log; production path anchored on FLock on-chain
        attestation.
      </p>
    </section>
  );
}

/**
 * "Verify" — fetches the signed Merkle root and shows size + root hash, the
 * tamper-evidence affordance. The signature presence is surfaced; we never
 * claim verification we didn't do.
 */
function VerifyControl() {
  const [root, setRoot] = useState<TransparencyRoot | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(false);

  const verify = useCallback(async (): Promise<void> => {
    setBusy(true);
    setErr(false);
    try {
      setRoot(await controlPlane.transparencyRoot());
    } catch {
      setErr(true);
      setRoot(null);
    } finally {
      setBusy(false);
    }
  }, []);

  if (root) {
    const shortRoot =
      root.rootHash.length > 18
        ? `${root.rootHash.slice(0, 10)}…${root.rootHash.slice(-6)}`
        : root.rootHash;
    return (
      <div
        className="shrink-0 rounded-[10px] border px-2.5 py-1.5 text-right"
        style={{ borderColor: "rgba(52,211,153,0.35)", background: "rgba(52,211,153,0.06)" }}
        title={`signed root ${root.rootHash}`}
      >
        <p className="eyebrow flex items-center justify-end gap-1.5" style={{ color: "var(--fed)" }}>
          <span aria-hidden>✓</span> root verified
        </p>
        <p className="tabular mt-1 text-[11px] text-text-secondary">
          size {root.size} · <span className="font-mono">{shortRoot}</span>
        </p>
        <p className="mt-0.5 text-[10px] text-text-muted">
          {root.signaturePem ? "Ed25519 signed" : "unsigned"}
        </p>
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={verify}
      disabled={busy}
      className="shrink-0 inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] transition-colors disabled:opacity-60"
      style={{
        background: "var(--bg-surface-2)",
        borderColor: err ? "var(--silo)" : "var(--border-default)",
        color: err ? "var(--silo)" : "var(--text-muted)",
      }}
    >
      <span aria-hidden className="h-1.5 w-1.5 rounded-full" style={{ background: "var(--accent-gold)" }} />
      {busy ? "verifying…" : err ? "plane offline · retry" : "verify · tamper-evident"}
    </button>
  );
}

function ProvenanceRow({ record }: { record: Provenance }) {
  const contributors = record.contributors ?? [];
  const rejected = record.rejected ?? [];
  const recallPct = Math.round((record.globalRecall ?? 0) * 100);

  return (
    <li
      className="rounded-[14px] border bg-bg-deep/50 p-3.5"
      style={{ borderColor: rejected.length > 0 ? "var(--silo-deep)" : "var(--border-default)" }}
    >
      <div className="flex items-center justify-between gap-3">
        <span
          className="tabular shrink-0 rounded-[8px] px-2 py-0.5 text-[11px] font-semibold tracking-wide text-bg-deep"
          style={{ background: "var(--accent-gold)" }}
        >
          R{record.round}
        </span>
        <div className="flex items-baseline gap-1.5">
          <span className="eyebrow">recall</span>
          <span
            className="tabular text-[15px] font-semibold"
            style={{ color: recallPct >= 75 ? "var(--fed)" : "var(--text-secondary)" }}
          >
            {recallPct}%
          </span>
        </div>
      </div>

      <div className="mt-2.5 flex flex-wrap items-center gap-1">
        {contributors.map((id) => (
          <BankTag key={id} id={id} variant="kept" />
        ))}
        {rejected.map((id) => (
          <BankTag key={id} id={id} variant="rejected" />
        ))}
      </div>
    </li>
  );
}

function BankTag({ id, variant }: { id: string; variant: "kept" | "rejected" }) {
  const isRejected = variant === "rejected";
  return (
    <span
      className="tabular inline-flex items-center gap-1 rounded-[7px] px-1.5 py-0.5 text-[10px] font-medium"
      style={
        isRejected
          ? { background: "rgba(240,74,82,0.16)", color: "var(--silo)" }
          : { background: "var(--bg-surface-2)", color: "var(--text-secondary)" }
      }
      title={isRejected ? `${id} rejected — poisoned update dropped` : `${id} contributed`}
    >
      {isRejected && (
        <span aria-hidden className="font-semibold" style={{ color: "var(--silo)" }}>
          ✕
        </span>
      )}
      {id}
    </span>
  );
}

function Legend({ source }: { source: "control-plane" | "node" }) {
  return (
    <div className="flex items-center gap-4 text-[11px] text-text-muted">
      <span className="flex items-center gap-1.5">
        <span aria-hidden className="h-2 w-2 rounded-[3px]" style={{ background: "var(--bg-surface-2)" }} />
        contributed
      </span>
      <span className="flex items-center gap-1.5">
        <span aria-hidden className="h-2 w-2 rounded-[3px]" style={{ background: "rgba(240,74,82,0.5)" }} />
        rejected
      </span>
      <span className="ml-auto text-[10px] uppercase tracking-[0.12em]">
        {source === "control-plane" ? "control plane" : "node"}
      </span>
    </div>
  );
}

function EmptyState({ offline }: { offline: boolean }) {
  return (
    <p
      className="rounded-[14px] border border-dashed p-4 text-[13px] leading-relaxed text-text-muted"
      style={{ borderColor: "var(--border-default)" }}
    >
      {offline
        ? "Control plane offline — no attested rounds available. The ledger will populate once the federation control plane is reachable."
        : "No rounds attested yet. Advance a federated round to begin recording the model bill-of-materials."}
    </p>
  );
}
