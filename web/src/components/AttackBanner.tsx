"use client";
import { useVeritas } from "../lib/store";

/** Transient banner driven by the store's `lastAttack` (attack_detected SSE). */
export default function AttackBanner() {
  const { lastAttack } = useVeritas();
  if (!lastAttack) return null;

  const rejected = lastAttack.rejected;
  return (
    <div
      role="status"
      className="rise flex items-center gap-3 rounded-[14px] border px-4 py-3"
      style={{
        borderColor: rejected ? "rgba(52,211,153,0.4)" : "var(--silo)",
        background: rejected ? "rgba(52,211,153,0.08)" : "rgba(240,74,82,0.1)",
      }}
    >
      <span
        className="grid h-7 w-7 place-items-center rounded-full text-[14px]"
        style={{
          background: rejected ? "var(--fed-deep)" : "var(--silo-deep)",
          color: rejected ? "var(--fed)" : "var(--silo)",
        }}
        aria-hidden
      >
        {rejected ? "✓" : "!"}
      </span>
      <div>
        <p className="text-[13px] font-semibold text-text-primary">
          Malicious update from {lastAttack.bankId} {rejected ? "rejected" : "detected"}
        </p>
        <p className="text-[12px] text-text-secondary">
          Multi-Krum + differential privacy filtered the poisoned gradients before aggregation.
        </p>
      </div>
    </div>
  );
}
