import type {
  State,
  Bank,
  Provenance,
  PredictRequest,
  PredictResponse,
} from "./types";

// NEXT_PUBLIC_API_BASE points at the Veritas CORE ENGINE (the federated
// aggregator), not an individual bank node. In local dev the engine runs on
// :8000 (see repo README); the standalone mock also serves the same routes on
// :8001 — point this var at whichever you started. A missing .env.local simply
// falls back to the documented engine port below.
const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

const post = (p: string, b?: unknown): Promise<unknown> =>
  fetch(`${BASE}${p}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(b ?? {}),
  }).then((r) => r.json());

export const api = {
  state: (): Promise<State> => fetch(`${BASE}/state`).then((r) => r.json()),
  banks: (): Promise<Bank[]> => fetch(`${BASE}/banks`).then((r) => r.json()),
  provenance: (): Promise<Provenance[]> =>
    fetch(`${BASE}/provenance`).then((r) => r.json()),
  step: (): Promise<State> => post("/round/step") as Promise<State>,
  reset: (): Promise<State> => post("/sim/reset") as Promise<State>,
  injectCampaign: (): Promise<unknown> =>
    post("/campaign/inject", { typology: "safe-account-mule" }),
  injectAttack: (memberId = "bank0"): Promise<unknown> =>
    post("/attack/inject", { memberId }),
  predict: (transaction: PredictRequest["transaction"]): Promise<PredictResponse> =>
    post("/predict", { transaction } satisfies PredictRequest) as Promise<PredictResponse>,
  eventsUrl: `${BASE}/events`,
};
