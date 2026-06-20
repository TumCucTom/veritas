"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { useVeritas } from "../lib/store";

const AUTO_RUN_MS = 1200;

export default function Controls() {
  const { state, refresh, connection } = useVeritas();
  const [autoRun, setAutoRun] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const run = useCallback(
    async (key: string, fn: () => Promise<unknown>): Promise<void> => {
      setBusy(key);
      try {
        await fn();
        await refresh();
      } catch {
        // Surface nothing destructive — connection chip already reflects offline.
      } finally {
        setBusy(null);
      }
    },
    [refresh],
  );

  // Auto-run loop: advance one federated round on a fixed cadence.
  useEffect(() => {
    if (!autoRun) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = null;
      return;
    }
    intervalRef.current = setInterval(() => {
      void api
        .step()
        .then(() => refresh())
        .catch(() => undefined);
    }, AUTO_RUN_MS);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = null;
    };
  }, [autoRun, refresh]);

  const campaignActive = state?.campaignActive ?? false;
  const round = state?.round ?? 0;

  return (
    <div
      className="flex flex-wrap items-center gap-2.5 rounded-[16px] border bg-bg-surface/70 p-2.5 backdrop-blur-sm sm:gap-3 sm:p-3"
      style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-raise)" }}
    >
      <button
        type="button"
        onClick={() => void run("campaign", api.injectCampaign)}
        disabled={busy !== null}
        aria-pressed={campaignActive}
        className="group relative inline-flex items-center gap-2 rounded-[11px] px-4 py-2.5 text-[13px] font-semibold text-bg-deep transition-all duration-200 ease-[cubic-bezier(0.16,1,0.3,1)] hover:-translate-y-px focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-gold focus-visible:ring-offset-2 focus-visible:ring-offset-bg-deep disabled:cursor-not-allowed disabled:opacity-50"
        style={{
          background: "linear-gradient(180deg, var(--accent-gold-soft), var(--accent-gold))",
          boxShadow: "0 8px 22px -10px rgba(214,168,91,0.7)",
        }}
      >
        <Dot />
        {campaignActive ? "Campaign live" : "Inject scam campaign"}
      </button>

      <ToggleButton active={autoRun} onClick={() => setAutoRun((v) => !v)} disabled={busy !== null}>
        {autoRun ? "Pause auto-run" : "Auto-run"}
      </ToggleButton>

      <GhostButton
        onClick={() => void run("step", api.step)}
        disabled={busy !== null || autoRun}
      >
        Step round
      </GhostButton>

      <DangerButton
        onClick={() => void run("attack", () => api.injectAttack("bank0"))}
        disabled={busy !== null}
      >
        Inject malicious member
      </DangerButton>

      <div className="ml-auto flex items-center gap-3 pl-1 pr-1">
        <span className="eyebrow">Round</span>
        <span className="tabular font-display text-lg leading-none text-text-primary">
          {String(round).padStart(2, "0")}
        </span>
        <ConnectionChip connection={connection} />
        <GhostButton
          onClick={() => {
            setAutoRun(false);
            void run("reset", api.reset);
          }}
          disabled={busy !== null}
        >
          Reset
        </GhostButton>
      </div>
    </div>
  );
}

function Dot() {
  return (
    <span
      aria-hidden
      className="inline-block h-1.5 w-1.5 rounded-full"
      style={{ background: "var(--bg-deep)" }}
    />
  );
}

const baseBtn =
  "inline-flex items-center gap-2 rounded-[11px] px-4 py-2.5 text-[13px] font-semibold transition-all duration-200 ease-[cubic-bezier(0.16,1,0.3,1)] hover:-translate-y-px focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-bg-deep disabled:cursor-not-allowed disabled:opacity-50";

function ToggleButton({
  active,
  children,
  ...rest
}: { active: boolean } & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      aria-pressed={active}
      className={`${baseBtn} border focus-visible:ring-fed`}
      style={{
        borderColor: active ? "var(--fed)" : "var(--border-strong)",
        background: active ? "rgba(52,211,153,0.12)" : "var(--bg-surface-2)",
        color: active ? "var(--fed)" : "var(--text-primary)",
      }}
      {...rest}
    >
      {active && (
        <span
          aria-hidden
          className="inline-block h-1.5 w-1.5 animate-pulse rounded-full"
          style={{ background: "var(--fed)" }}
        />
      )}
      {children}
    </button>
  );
}

function GhostButton(props: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className={`${baseBtn} border border-border-strong bg-transparent text-text-secondary hover:border-border-strong hover:bg-bg-surface-2 hover:text-text-primary focus-visible:ring-accent-gold`}
      {...props}
    />
  );
}

function DangerButton(props: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className={`${baseBtn} border focus-visible:ring-silo`}
      style={{
        borderColor: "rgba(240,74,82,0.4)",
        background: "rgba(240,74,82,0.08)",
        color: "var(--silo)",
      }}
      {...props}
    />
  );
}

function ConnectionChip({
  connection,
}: {
  connection: "connecting" | "live" | "offline";
}) {
  const map = {
    connecting: { label: "connecting", color: "var(--accent-gold)" },
    live: { label: "live", color: "var(--fed)" },
    offline: { label: "offline", color: "var(--text-muted)" },
  } as const;
  const { label, color } = map[connection];
  return (
    <span className="flex items-center gap-1.5">
      <span
        aria-hidden
        className="inline-block h-1.5 w-1.5 rounded-full"
        style={{ background: color, boxShadow: `0 0 8px ${color}` }}
      />
      <span className="eyebrow" style={{ color }}>
        {label}
      </span>
    </span>
  );
}
