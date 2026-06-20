"""Veritas Tier 2 Federation Control Plane — FastAPI app (port :9000).

Run: uvicorn controlplane.server.app:app --port 9000

Implements PROTOCOL.md "Tier 2 — Control Plane API": enrolment & identity,
federation rounds (Multi-Krum aggregation), model registry & governance,
the signed Merkle transparency log, and the per-tenant console + SSE events.
"""
from __future__ import annotations

import asyncio
import os

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..state import ControlPlane
from ..transparency import verify_inclusion_proof  # noqa: F401 (re-exported for tests)

ADMIN_KEY = os.environ.get("VERITAS_ADMIN_KEY", "dev-admin-key")
# Auto-aggregation threshold. Deployments/integration tests can raise this so
# that aggregation is driven explicitly via POST /v1/rounds/advance instead of
# firing the moment `minUpdates` honest updates arrive.
_MIN_UPDATES = int(os.environ.get("VERITAS_MIN_UPDATES", "3"))

app = FastAPI(title="Veritas Control Plane", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Module-level plane so tests can import & inspect it; rebuildable via reset.
plane = ControlPlane(admin_key=ADMIN_KEY, min_updates=_MIN_UPDATES)


def get_plane() -> ControlPlane:
    return plane


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------
def require_admin(x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")):
    if x_admin_key != plane.admin_key:
        raise HTTPException(status_code=401, detail="invalid admin key")


def require_member(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        return plane.authenticate(token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail=f"auth failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class EnrollBody(BaseModel):
    memberId: str
    displayName: str
    publicKeyPem: str
    # Opaque attestation quote: a string (tests/CLI) or the node's structured
    # TEE/software-stub quote dict (Quote.to_wire()). Stored verbatim.
    attestationQuote: dict | str | None = None


class UpdateBody(BaseModel):
    memberId: str
    round: int
    update: list[float]
    numExamples: int
    localMetrics: dict = Field(default_factory=dict)
    attestationQuote: dict | str | None = None


# ---------------------------------------------------------------------------
# Enrolment & identity
# ---------------------------------------------------------------------------
@app.post("/v1/members/enroll", status_code=201)
def enroll(body: EnrollBody):
    try:
        m = plane.enroll(body.memberId, body.displayName, body.publicKeyPem,
                         body.attestationQuote)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # invalid PEM etc.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"memberId": m.member_id, "tenantId": m.tenant_id, "status": m.status}


@app.post("/v1/members/{member_id}/approve", dependencies=[Depends(require_admin)])
def approve(member_id: str):
    try:
        m = plane.approve(member_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown member") from exc
    return {"memberId": m.member_id, "status": m.status}


@app.get("/v1/members", dependencies=[Depends(require_admin)])
def list_members():
    return [m.public() for m in plane.members.values()]


# ---------------------------------------------------------------------------
# Federation rounds
# ---------------------------------------------------------------------------
@app.get("/v1/rounds/current")
def current_round():
    return plane.current_round_info()


@app.get("/v1/models/current")
def current_model():
    return plane.current_model().public()


@app.get("/v1/models/registry")
def registry():
    return plane.registry()


@app.get("/v1/models/{version}")
def get_model(version: int):
    try:
        return plane.get_model(version).public()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown version") from exc


@app.post("/v1/rounds/{round_}/updates", status_code=202)
def submit_update(round_: int, body: UpdateBody, member=Depends(require_member)):
    if body.memberId != member.member_id:
        raise HTTPException(status_code=403, detail="memberId mismatch with token")
    try:
        res = plane.submit_update(
            body.memberId, round_, body.update, body.numExamples,
            body.localMetrics, body.attestationQuote)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Auto-aggregate when enough updates have arrived.
    if res.pop("_ready", False):
        plane.maybe_aggregate(round_)
    return res


@app.post("/v1/rounds/advance", dependencies=[Depends(require_admin)])
def advance_round():
    result = plane.advance_round()
    if result is None:
        return {"aggregated": False, "round": plane.current_round}
    return {"aggregated": True, "round": result.round,
            "newVersion": result.new_version}


@app.get("/v1/rounds/{round_}/result")
def round_result(round_: int):
    try:
        return plane.get_round_result(round_)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="no result for round") from exc


# ---------------------------------------------------------------------------
# Model governance
# ---------------------------------------------------------------------------
@app.post("/v1/models/{version}/promote", dependencies=[Depends(require_admin)])
def promote(version: int):
    try:
        return plane.promote(version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown version") from exc


@app.post("/v1/models/{version}/rollback", dependencies=[Depends(require_admin)])
def rollback(version: int):
    try:
        return plane.rollback(version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown version") from exc


# ---------------------------------------------------------------------------
# Transparency log
# ---------------------------------------------------------------------------
@app.get("/v1/transparency")
def transparency_list():
    return plane.transparency.list()


@app.get("/v1/transparency/root")
def transparency_root():
    return plane.transparency.signed_root()


@app.get("/v1/transparency/proof/{seq}")
def transparency_proof(seq: int):
    try:
        return plane.transparency.inclusion_proof(seq)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail="seq out of range") from exc


# ---------------------------------------------------------------------------
# Tenant console
# ---------------------------------------------------------------------------
@app.get("/v1/tenants/{tenant_id}/state")
def tenant_state(tenant_id: str):
    return plane.tenant_state(tenant_id)


@app.get("/v1/tenants/{tenant_id}/provenance")
def tenant_provenance(tenant_id: str):
    return plane.tenant_provenance(tenant_id)


@app.get("/v1/events")
async def events(request: Request, tenantId: str | None = None):
    queue: asyncio.Queue = asyncio.Queue()
    plane.subscribe(queue)

    async def gen():
        try:
            # Initial comment to open the stream promptly.
            yield {"event": "ping", "data": "connected"}
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "keepalive"}
                    continue
                import json
                yield {"event": msg["event"], "data": json.dumps(msg["data"])}
        finally:
            plane.unsubscribe(queue)

    return EventSourceResponse(gen())


@app.get("/health")
def health():
    return {"ok": True, "round": plane.current_round,
            "modelVersion": plane.promoted_version,
            "members": len(plane.members)}
