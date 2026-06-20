"use client";
import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import { useEvents } from "../lib/useEvents";
import type { Provenance, VeritasEvent } from "../lib/types";

export default function ProvenancePanel() {
  const [records, setRecords] = useState<Provenance[]>([]);

  const load = useCallback(async (): Promise<void> => {
    try {
      const next = await api.provenance();
      setRecords(Array.isArray(next) ? next : []);
    } catch {
      // Provenance endpoint unreachable — keep the last known ledger.
    }
  }, []);

  // Initial fetch, then re-pull the ledger whenever a round closes so the
  // bill-of-materials stays in lockstep with the global model on screen.
  useEffect(() => {
    void load();
  }, [load]);

  const onEvent = useCallback(
    (e: VeritasEvent): void => {
      if (e.type === "round_complete") void load();
    },
    [load],
  );
  useEvents(onEvent);

  const rows = records ?? [];

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
        <span
          className="hidden shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.12em] sm:inline-flex"
          style={{ background: "var(--bg-surface-2)", color: "var(--text-muted)" }}
        >
          <span
            aria-hidden
            className="h-1.5 w-1.5 rounded-full"
            style={{ background: "var(--accent-gold)" }}
          />
          tamper-evident
        </span>
      </header>

      {rows.length === 0 ? (
        <EmptyState />
      ) : (
        <div className="flex flex-col gap-2.5">
          <Legend />
          <ol className="flex flex-col gap-2">
            {rows.map((rec) => (
              <ProvenanceRow key={rec.round} record={rec} />
            ))}
          </ol>
        </div>
      )}

      <p className="mt-auto pt-5 text-[11px] leading-snug text-text-muted">
        Production: each round anchored on FLock on-chain attestation.
      </p>
    </section>
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

function Legend() {
  return (
    <div className="flex items-center gap-4 text-[11px] text-text-muted">
      <span className="flex items-center gap-1.5">
        <span
          aria-hidden
          className="h-2 w-2 rounded-[3px]"
          style={{ background: "var(--bg-surface-2)" }}
        />
        contributed
      </span>
      <span className="flex items-center gap-1.5">
        <span
          aria-hidden
          className="h-2 w-2 rounded-[3px]"
          style={{ background: "rgba(240,74,82,0.5)" }}
        />
        rejected
      </span>
    </div>
  );
}

function EmptyState() {
  return (
    <p className="rounded-[14px] border border-dashed p-4 text-[13px] leading-relaxed text-text-muted"
      style={{ borderColor: "var(--border-default)" }}
    >
      No rounds attested yet. Inject a campaign and advance a federated round to
      begin recording the model bill-of-materials.
    </p>
  );
}
