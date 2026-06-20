"use client";
import { useVeritas } from "../lib/store";
import { formatNumber } from "../lib/format";

/**
 * The privacy headline: raw customer records that left the bank.
 * It is engineered to read 0 — that is the federated story.
 */
export default function PrivacyBadge() {
  const { state } = useVeritas();
  const records = state?.customerRecordsTransmitted ?? 0;
  const clean = records === 0;

  return (
    <div
      className="flex items-center gap-3 rounded-full border px-4 py-2"
      style={{
        borderColor: clean ? "rgba(52,211,153,0.35)" : "var(--silo)",
        background: clean ? "rgba(52,211,153,0.06)" : "rgba(240,74,82,0.08)",
      }}
    >
      <span className="relative flex h-2 w-2">
        <span
          className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60"
          style={{ background: clean ? "var(--fed)" : "var(--silo)" }}
        />
        <span
          className="relative inline-flex h-2 w-2 rounded-full"
          style={{ background: clean ? "var(--fed)" : "var(--silo)" }}
        />
      </span>
      <span className="text-[11px] uppercase tracking-[0.16em] text-text-muted">
        Customer records transmitted
      </span>
      <span
        className="tabular font-display text-lg leading-none"
        style={{ color: clean ? "var(--fed)" : "var(--silo)" }}
      >
        {formatNumber(records)}
      </span>
    </div>
  );
}
