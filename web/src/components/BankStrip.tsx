"use client";
import { useVeritas } from "../lib/store";
import type { Bank } from "../lib/types";

export default function BankStrip() {
  const { state, lastAttack } = useVeritas();
  const banks = state?.banks ?? [];

  return (
    <section
      aria-labelledby="bankstrip-heading"
      className="rounded-[20px] border bg-bg-surface/70 p-5 backdrop-blur-sm sm:p-6"
      style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-raise)" }}
    >
      <header className="mb-4 flex items-baseline justify-between gap-4">
        <div>
          <p className="eyebrow">Network members</p>
          <h2
            id="bankstrip-heading"
            className="font-display text-xl tracking-tight text-text-primary"
          >
            Eight UK institutions
          </h2>
        </div>
        <Legend />
      </header>

      {banks.length === 0 ? (
        <SkeletonRow />
      ) : (
        <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-8">
          {banks.map((bank) => (
            <BankChip
              key={bank.id}
              bank={bank}
              flashing={lastAttack?.bankId === bank.id}
              rejected={lastAttack?.bankId === bank.id ? lastAttack.rejected : false}
            />
          ))}
        </ul>
      )}
    </section>
  );
}

function BankChip({
  bank,
  flashing,
  rejected,
}: {
  bank: Bank;
  flashing: boolean;
  rejected: boolean;
}) {
  const fed = Math.round((bank.detection?.federated ?? 0) * 100);
  const silo = Math.round((bank.detection?.siloed ?? 0) * 100);
  const poisoned = bank.poisoned;

  return (
    <li
      className={`relative rounded-[14px] border bg-bg-surface-2 p-3 transition-colors ${flashing ? "attack-flash" : ""}`}
      style={{
        borderColor: poisoned ? "var(--silo)" : "var(--border-default)",
      }}
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="truncate text-[13px] font-semibold text-text-primary">{bank.name}</span>
        {poisoned ? (
          <span
            className="shrink-0 rounded-full px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-[0.1em]"
            style={{ background: "rgba(240,74,82,0.16)", color: "var(--silo)" }}
          >
            poisoned
          </span>
        ) : (
          <span
            aria-hidden
            className="h-1.5 w-1.5 shrink-0 rounded-full"
            style={{ background: "var(--fed)", opacity: fed > 70 ? 1 : 0.25 }}
          />
        )}
      </div>

      <Bar label="Fed" pct={fed} color="var(--fed)" />
      <Bar label="Solo" pct={silo} color="var(--silo)" />

      {flashing && (
        <span
          className="absolute -top-2 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-full px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.12em]"
          style={{
            background: rejected ? "var(--fed-deep)" : "var(--silo-deep)",
            color: rejected ? "var(--fed)" : "var(--silo)",
          }}
        >
          {rejected ? "update rejected" : "update accepted"}
        </span>
      )}
    </li>
  );
}

function Bar({ label, pct, color }: { label: string; pct: number; color: string }) {
  return (
    <div className="mb-1.5 flex items-center gap-2 last:mb-0">
      <span className="w-7 text-[9px] uppercase tracking-[0.08em] text-text-muted">{label}</span>
      <span
        className="relative h-1.5 flex-1 overflow-hidden rounded-full"
        style={{ background: "var(--bg-deep)" }}
      >
        <span
          className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-700 ease-[cubic-bezier(0.16,1,0.3,1)]"
          style={{ width: `${Math.max(2, pct)}%`, background: color }}
        />
      </span>
      <span className="tabular w-7 text-right text-[10px] text-text-secondary">{pct}%</span>
    </div>
  );
}

function Legend() {
  return (
    <div className="hidden items-center gap-3 sm:flex">
      <LegendDot color="var(--fed)" label="Federated" />
      <LegendDot color="var(--silo)" label="Siloed" />
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span aria-hidden className="h-2 w-2 rounded-full" style={{ background: color }} />
      <span className="text-[11px] text-text-muted">{label}</span>
    </span>
  );
}

function SkeletonRow() {
  return (
    <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-8">
      {Array.from({ length: 8 }).map((_, i) => (
        <li
          key={i}
          className="h-[78px] animate-pulse rounded-[14px] border"
          style={{ borderColor: "var(--border-default)", background: "var(--bg-surface-2)" }}
        />
      ))}
    </ul>
  );
}
