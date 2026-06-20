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

All state is **in-memory** in a single `ControlPlane` object (`state.py`):
members, model registry, rounds/updates, results, and the transparency log.
This is acceptable for the reference impl. It is structured behind one object
with typed records (`Member`, `Model`, `Update`, `RoundResult`) so a
persistence layer (Postgres + an append-only store for the log) can slot in by
swapping the dict-backed collections for repositories without touching the HTTP
layer in `server/app.py`.

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
the plane stacks the round's deltas and runs
`core.veritas_core.aggregation.multi_krum(deltas, n_byzantine=1, m=n-1)`.
Selected members are `contributors`; the rest are `rejected` and each fires an
`attack_detected` SSE event. The aggregate delta is added to the promoted
(parent) model -> a new **canary** version. A `round_aggregated` record is
appended to the transparency log, and the next round opens.

## Endpoints

Identity: `POST /v1/members/enroll`, `POST /v1/members/{id}/approve` (admin),
`GET /v1/members` (admin).
Rounds: `GET /v1/rounds/current`, `POST /v1/rounds/{round}/updates` (auth),
`POST /v1/rounds/advance` (admin), `GET /v1/rounds/{round}/result`.
Models: `GET /v1/models/current`, `GET /v1/models/{version}`,
`GET /v1/models/registry`, `POST /v1/models/{version}/promote` (admin),
`POST /v1/models/{version}/rollback` (admin).
Transparency: `GET /v1/transparency`, `GET /v1/transparency/root`,
`GET /v1/transparency/proof/{seq}`.
Tenant console: `GET /v1/tenants/{tenantId}/state`,
`GET /v1/tenants/{tenantId}/provenance`, `GET /v1/events?tenantId=...` (SSE:
`round_complete`, `model_promoted`, `attack_detected`, `member_enrolled`).
`GET /health`.

## Deviations from the protocol

- `tenantId` is derived as `tenant-<memberId>` (one tenant per member) in this
  reference impl; a real deployment would map members to bank tenants.
- The tenant `siloed` lift is modelled as 0.6× the federated recall (honest
  counterfactual, same shape as the hackathon engine), since the plane only
  receives metrics, not raw eval data.
- `dpParams` are reported for the node's client-side DP; the plane verifies
  Multi-Krum robustness rather than re-clipping (DP is applied node-side per the
  protocol's federation-client description).
