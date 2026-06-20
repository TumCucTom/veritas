import type { State, Bank } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8001";

const post = (p: string, b?: unknown): Promise<unknown> =>
  fetch(`${BASE}${p}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(b ?? {}),
  }).then((r) => r.json());

export const api = {
  state: (): Promise<State> => fetch(`${BASE}/state`).then((r) => r.json()),
  banks: (): Promise<Bank[]> => fetch(`${BASE}/banks`).then((r) => r.json()),
  step: (): Promise<State> => post("/round/step") as Promise<State>,
  reset: (): Promise<State> => post("/sim/reset") as Promise<State>,
  injectCampaign: (): Promise<unknown> =>
    post("/campaign/inject", { typology: "safe-account-mule" }),
  injectAttack: (memberId = "bank0"): Promise<unknown> =>
    post("/attack/inject", { memberId }),
  predict: (transaction: Record<string, number>): Promise<unknown> =>
    post("/predict", { transaction }),
  eventsUrl: `${BASE}/events`,
};
