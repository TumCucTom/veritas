/*
 * Control-plane (Tier 2) API client.
 *
 * The production console reads from TWO data sources that coexist:
 *   - the bank NODE (Tier 1) via NEXT_PUBLIC_API_BASE — the live race / hero /
 *     inspector (handled by ./api.ts), and
 *   - the federation CONTROL PLANE (Tier 2) via NEXT_PUBLIC_CONTROL_PLANE —
 *     federation, governance and provenance for one tenant (this module).
 *
 * Every call degrades gracefully: a fetch failure throws, and callers catch it
 * to render a "control plane offline" state rather than crash the page.
 */
import type {
  ModelRegistryEntry,
  TenantProvenance,
  TenantState,
  TransparencyEntry,
  TransparencyRoot,
} from "./controlplane-types";

export const CONTROL_PLANE_BASE =
  process.env.NEXT_PUBLIC_CONTROL_PLANE ?? "http://localhost:9000";

export const TENANT_ID = process.env.NEXT_PUBLIC_TENANT_ID ?? "tenant-demo";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${CONTROL_PLANE_BASE}${path}`, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) throw new Error(`control plane ${path} -> ${res.status}`);
  return (await res.json()) as T;
}

export const controlPlane = {
  base: CONTROL_PLANE_BASE,
  tenantId: TENANT_ID,

  tenantState: (tid: string = TENANT_ID): Promise<TenantState> =>
    getJson<TenantState>(`/v1/tenants/${encodeURIComponent(tid)}/state`),

  tenantProvenance: (tid: string = TENANT_ID): Promise<TenantProvenance[]> =>
    getJson<TenantProvenance[]>(
      `/v1/tenants/${encodeURIComponent(tid)}/provenance`,
    ),

  transparencyRoot: (): Promise<TransparencyRoot> =>
    getJson<TransparencyRoot>(`/v1/transparency/root`),

  transparency: (): Promise<TransparencyEntry[]> =>
    getJson<TransparencyEntry[]>(`/v1/transparency`),

  modelRegistry: (): Promise<ModelRegistryEntry[]> =>
    getJson<ModelRegistryEntry[]>(`/v1/models/registry`),

  /** SSE endpoint for tenant-scoped named events. */
  eventsUrl: (tid: string = TENANT_ID): string =>
    `${CONTROL_PLANE_BASE}/v1/events?tenantId=${encodeURIComponent(tid)}`,
};
