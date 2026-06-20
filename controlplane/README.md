# Veritas — Tier 2 Federation Control Plane

Reference implementation of the **Federation Control Plane** (`:9000`) from
`protocol/PROTOCOL.md` and the production architecture spec. Python / FastAPI.

It coordinates federation rounds across bank nodes, robustly aggregates their
DP-protected model deltas with **Multi-Krum** (imported from
`core/veritas_core`, never reimplemented), governs the global model registry
(canary / promote / rollback), and emits tamper-evident provenance via a
signed **Merkle transparency log**. It also serves the per-tenant console API
and an SSE event stream.

## Run

```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install -e .            # the control plane
pip install -e ../core      # FL primitives: `from veritas_core...`
uvicorn controlplane.server.app:app --port 9000
```

`VERITAS_ADMIN_KEY` (default `dev-admin-key`) sets the `X-Admin-Key` admin secret.

## Tests

```bash
. .venv/bin/activate
python -m pytest -q
```

## State / persistence

Working state lives in memory in a single `ControlPlane` object (`state.py`) for
speed, but every write is **persisted write-through to SQLite** (`store.py`,
stdlib `sqlite3`): members, model registry, rounds/results, the transparency
log, and the DP-spent counter all survive a restart. On start the plane
load-rehydrates from the DB (genesis-seeds if empty).

- `VERITAS_DB` (default `controlplane.db`; `:memory:` in tests) — DB file path.
- `VERITAS_PLANE_KEY` (default `plane_key.pem`; `:memory:` in tests) — the
  control-plane Ed25519 **signing identity** is persisted (PEM, `0600`,
  generate-if-absent) so the transparency log stays verifiable across restarts.

## Auth (for Tier 1 node authors)

- **Enrolment:** `POST /v1/members/enroll` with `{memberId, displayName,
  publicKeyPem, attestationQuote?}`. `publicKeyPem` is an **Ed25519** SPKI PEM.
  Returns `status: "pending"`. An admin then calls
  `POST /v1/members/{memberId}/approve` (`X-Admin-Key`) -> `active`.
- **Node -> plane auth:** every authenticated request carries
  `Authorization: Bearer <jwt>`. The JWT is **EdDSA**-signed by the member's
  **private** key with claims `{sub: memberId, tid: tenantId, iat, exp}`. The
  plane verifies against the registered public key; a pending/unknown member or
  bad signature -> `401`. `controlplane/crypto.py::make_member_jwt` is the
  reference signer.
- **Admin endpoints** require `X-Admin-Key`.

## Aggregation

On `#updates >= minUpdates` (default 3) — or admin `POST /v1/rounds/advance` —
the plane robustly aggregates the round's DP-noised deltas using the principled
core primitive `veritas_core.robust.robust_aggregate` (imported, never
reimplemented). Two stages:

- **Detection** — Multi-Krum (`method='krum'`) yields per-update Krum scores;
  the dropped candidate is reported as `rejected` (and fires `attack_detected`)
  **only** when its score dwarfs the honest cloud (robust ratio gate), so an
  all-honest round accuses no one while a genuine poisoner is named.
- **Aggregate** — the configured method (`VERITAS_AGG_METHOD`, default
  `trimmed_mean`) over the surviving updates, with `max_norm` clipping
  amplification, produces the global delta added to the promoted parent model
  → a new **canary** version. A `round_aggregated` record is appended to the
  transparency log, the DP budget advances, and the next round opens.

Env: `VERITAS_AGG_METHOD` (`trimmed_mean`|`median`|`krum`|`bulyan`, default
`trimmed_mean`), `VERITAS_N_BYZANTINE` (assumed Byzantine count `f`, default 1).

## Differential privacy

At startup the plane calibrates a noise multiplier with
`veritas_core.dp_accountant.calibrate_noise_multiplier(target_epsilon, delta,
num_rounds)` and advertises it in `dpParams.sigma` (with `epsilon`/`delta`).
Nodes read `dpParams.sigma` to noise their client-side updates. A long-lived
`RDPAccountant` tracks SPENT epsilon (one `.step(sigma)` per aggregated round),
exposed at `GET /v1/privacy`.

Env: `VERITAS_DP_EPSILON` (default `8.0`), `VERITAS_DP_DELTA` (default `1e-5`),
`VERITAS_DP_ROUNDS` (default `50`). Observed calibration: sigma ≈ 4.51, spent
epsilon ≈ 8.0 after 50 rounds.

## Endpoints

Identity: `POST /v1/members/enroll`, `POST /v1/members/{id}/approve` (admin),
`GET /v1/members` (admin).
Rounds: `GET /v1/rounds/current`, `POST /v1/rounds/{round}/updates` (auth),
`POST /v1/rounds/advance` (admin), `GET /v1/rounds/{round}/result`.
Models: `GET /v1/models/current`, `GET /v1/models/{version}`,
`GET /v1/models/registry`, `POST /v1/models/{version}/promote` (admin),
`POST /v1/models/{version}/rollback` (admin).
Transparency: `GET /v1/transparency`, `GET /v1/transparency/root`,
`GET /v1/transparency/proof/{seq}` (inclusion),
`GET /v1/transparency/consistency?first=&second=` (RFC-6962-style append-only
consistency proof between two tree sizes).
Privacy: `GET /v1/privacy` -> `{targetEpsilon, delta, spentEpsilon,
noiseMultiplier, roundsSpent, budgetRemaining}`.
Tenant console: `GET /v1/tenants/{tenantId}/state`,
`GET /v1/tenants/{tenantId}/provenance`, `GET /v1/events?tenantId=...` (SSE:
`round_complete`, `model_promoted`, `attack_detected`, `member_enrolled`).
`GET /health`.

## Deviations from the protocol

- `tenantId` is derived as `tenant-<memberId>` (one tenant per member) in this
  reference impl; a real deployment would map members to bank tenants.
- The tenant `siloed` lift uses the **measured** siloed-baseline recall the
  nodes report each round (`localMetrics.siloRecall`, averaged over rounds that
  reported it; `lift.siloed.source == "reported"`). When no node has reported it
  yet (older nodes) it falls back to the prior 0.6× heuristic
  (`source == "heuristic"`).
- `dpParams` are reported for the node's client-side DP; the plane verifies
  Multi-Krum robustness rather than re-clipping (DP is applied node-side per the
  protocol's federation-client description).
