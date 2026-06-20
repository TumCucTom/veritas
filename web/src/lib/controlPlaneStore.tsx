"use client";
/*
 * Control-plane data provider — the Tier 2 half of the console.
 *
 * Loads the per-tenant state, provenance ledger and model registry from the
 * control plane, and listens to the tenant SSE stream to refresh on round /
 * promotion events. It degrades gracefully: if the control plane is
 * unreachable, `status` becomes "offline" and the panels render an explicit
 * offline state instead of crashing.
 *
 * Kept independent from ./store.tsx (the node race) so the two sources coexist.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { controlPlane, TENANT_ID } from "./controlplane";
import type {
  ModelRegistryEntry,
  TenantProvenance,
  TenantState,
} from "./controlplane-types";

export type ControlPlaneStatus = "connecting" | "live" | "offline";

interface ControlPlaneData {
  status: ControlPlaneStatus;
  tenantId: string;
  state: TenantState | null;
  provenance: TenantProvenance[];
  registry: ModelRegistryEntry[];
  refresh: () => Promise<void>;
}

const Ctx = createContext<ControlPlaneData | null>(null);

export function ControlPlaneProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<ControlPlaneStatus>("connecting");
  const [state, setState] = useState<TenantState | null>(null);
  const [provenance, setProvenance] = useState<TenantProvenance[]>([]);
  const [registry, setRegistry] = useState<ModelRegistryEntry[]>([]);
  const mounted = useRef(true);

  const refresh = useCallback(async (): Promise<void> => {
    // Each source is settled independently: a missing registry endpoint should
    // not blank out tenant state. We only flip to "offline" if state — the
    // primary source — fails.
    const [s, p, r] = await Promise.allSettled([
      controlPlane.tenantState(),
      controlPlane.tenantProvenance(),
      controlPlane.modelRegistry(),
    ]);
    if (!mounted.current) return;

    if (s.status === "fulfilled") {
      setState(s.value);
      setStatus("live");
    } else {
      setStatus("offline");
    }
    if (p.status === "fulfilled") setProvenance(Array.isArray(p.value) ? p.value : []);
    if (r.status === "fulfilled") setRegistry(Array.isArray(r.value) ? r.value : []);
  }, []);

  useEffect(() => {
    mounted.current = true;
    void refresh();
    return () => {
      mounted.current = false;
    };
  }, [refresh]);

  // Tenant SSE: refresh the slow-moving panels when the federation advances.
  useEffect(() => {
    let es: EventSource | null = null;
    try {
      es = new EventSource(controlPlane.eventsUrl(TENANT_ID));
    } catch {
      return; // SSE unsupported / bad URL — polling-on-mount still gave us data.
    }
    const onAny = (): void => void refresh();
    const names = [
      "round_complete",
      "model_promoted",
      "attack_detected",
      "member_enrolled",
    ] as const;
    for (const n of names) es.addEventListener(n, onAny);
    // If the stream errors we keep the last-known data; status is driven by refresh().
    return () => {
      for (const n of names) es?.removeEventListener(n, onAny);
      es?.close();
    };
  }, [refresh]);

  return (
    <Ctx.Provider
      value={{ status, tenantId: TENANT_ID, state, provenance, registry, refresh }}
    >
      {children}
    </Ctx.Provider>
  );
}

export function useControlPlane(): ControlPlaneData {
  const ctx = useContext(Ctx);
  if (!ctx)
    throw new Error("useControlPlane must be used within a ControlPlaneProvider");
  return ctx;
}
