"""Tier 1 Bank Node — FastAPI application.

Outward contract (the bank's systems call these): POST /predict, GET /state,
GET /events (SSE), GET /health. Edge endpoints (Tier 0 device side): GET
/edge/v1/model, POST /edge/v1/updates, POST /edge/v1/score. Plus dev controls
to drive the federation loop and the campaign toggle.

Run (node i = port 8100+i):
  VERITAS_NODE_INDEX=0 VERITAS_PLANE_URL=http://localhost:9000 \
      uvicorn node.server.app:app --port 8100
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ..config import NodeConfig
from ..predict_features import transaction_to_features
from ..runtime import NodeRuntime

cfg = NodeConfig.from_env()
runtime = NodeRuntime(cfg)

# SSE fan-out: the runtime emits events; we push them to subscriber queues.
_subs: list[asyncio.Queue] = []
_main_loop: asyncio.AbstractEventLoop | None = None


def _publish(ev: dict) -> None:
    loop = _main_loop
    if loop is None:
        return
    for q in list(_subs):
        loop.call_soon_threadsafe(q.put_nowait, ev)


runtime.set_event_sink(_publish)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    # Register identity up-front (off the event loop) so the member shows up as
    # `pending` and an admin can approve it before any federation round runs —
    # even when the federation loop is not autostarted.
    await asyncio.to_thread(runtime.ensure_enrolled)
    if cfg.autostart_federation:
        runtime.start_federation()
    yield
    runtime.stop_federation()


app = FastAPI(title=f"Veritas Bank Node — {cfg.node_id}", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---- models ---------------------------------------------------------------

class PredictBody(BaseModel):
    transaction: dict = {}
    text: str | None = None


class EdgeUpdateBody(BaseModel):
    deviceToken: str
    update: list[float]            # MASKED (clipped+DP-noised on-device, then masked)
    numExamples: int = 1
    cohortId: str | None = None    # secure-agg cohort/round id
    clientId: str | None = None    # device id within the cohort (mask cancellation)


class EdgeScoreBody(BaseModel):
    features: list[float]


class CohortOpenBody(BaseModel):
    clientIds: list[str]
    cohortId: str | None = None


class CohortCloseBody(BaseModel):
    cohortId: str | None = None


# ---- outward contract -----------------------------------------------------

@app.post("/predict")
def predict(body: PredictBody) -> dict:
    x = transaction_to_features(body.transaction)
    label, conf = runtime.engine.predict(x)
    indicators = []
    if x[-1] > 0:
        indicators.append("matches active campaign signature")
    indicators += ["new account", "high-velocity fan-out to multiple recipients"]
    return {
        "label": label,
        "confidence": round(conf, 3),
        "indicators": indicators,
        "explanation": "Scored locally on the federated global model; no customer data left the node.",
    }


@app.get("/state")
def state() -> dict:
    return runtime.engine.state()


@app.get("/health")
def health() -> dict:
    return runtime.health()


@app.get("/events")
async def events():
    q: asyncio.Queue = asyncio.Queue()
    _subs.append(q)

    async def gen():
        try:
            yield {"event": "round_complete", "data": json.dumps(runtime.engine.state())}
            while True:
                ev = await q.get()
                yield {"event": ev["type"], "data": json.dumps(ev["data"])}
        finally:
            if q in _subs:
                _subs.remove(q)

    return EventSourceResponse(gen())


# ---- edge endpoints (Tier 0 device side) ----------------------------------

@app.get("/edge/v1/model")
def edge_model() -> dict:
    return runtime.engine.edge_model()


@app.post("/edge/v1/cohort/open")
def edge_cohort_open(body: CohortOpenBody) -> dict:
    # Open a secure-aggregation cohort and deal pairwise seeds. Production: the
    # node only relays device X25519 public keys; seeds are derived via DH so
    # the node never learns them. Reference: trusted-dealer seeds (hex-encoded).
    table = runtime.engine.open_cohort(body.clientIds, cohort_id=body.cohortId)
    table["seedTable"] = {k: v.hex() for k, v in table["seedTable"].items()}
    return table


def _validate_vector(values: list[float], expected_dim: int, *, what: str) -> np.ndarray:
    """Parse a wire float vector, rejecting wrong length and non-finite values.

    Devices/SDKs are untrusted input: a wrong-dimension or NaN/inf vector must be
    rejected with a clear 422 BEFORE it reaches training / DP / aggregation,
    where it would silently poison the edge model (NaN propagates through the
    masked sum) or crash deep in numpy.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != expected_dim:
        raise HTTPException(
            status_code=422,
            detail=f"{what} must have length {expected_dim}, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise HTTPException(status_code=422, detail=f"{what} contains non-finite values")
    return arr


@app.post("/edge/v1/updates")
def edge_updates(body: EdgeUpdateBody) -> dict:
    # The device sends a MASKED update (clipped + DP-noised on-device, then
    # Bonawitz pairwise-masked). The node only buffers the masked message for
    # the cohort and can recover the aggregate SUM at close — it NEVER sees,
    # stores, or can derive any individual device's cleartext update.
    # deviceToken is ephemeral, not a customer identifier.
    #
    # The MASKED vector is still the edge-model dimension (masking is additive),
    # so we validate length + finiteness here. A masked value is large (mask
    # scale) but always finite; NaN/inf or a wrong length is a malformed/hostile
    # message and is rejected before it can poison the secure sum.
    update = _validate_vector(body.update, runtime.cfg.dim, what="edge update")
    runtime.engine.edge_aggregate(
        update, body.numExamples,
        cohort_id=body.cohortId, client_id=body.clientId,
    )
    return {"accepted": True}


@app.post("/edge/v1/cohort/close")
def edge_cohort_close(_body: CohortCloseBody | None = None) -> dict:
    # Secure-sum the cohort's masked messages (recovering any dropouts) and fold
    # the single aggregate into the edge model.
    runtime.engine.close_cohort()
    return {"accepted": True, "version": runtime.engine.global_version}


@app.post("/edge/v1/score")
def edge_score(body: EdgeScoreBody) -> dict:
    # /edge/v1/score scores a single FEATURE_DIM feature row (no bias term).
    # Validate length + finiteness before scoring: a wrong-dimension or NaN/inf
    # feature row is hostile/malformed input and must be rejected with a clear
    # 422 rather than crash deep in the model forward pass.
    x = _validate_vector(body.features, runtime.cfg.dim - 1, what="features")
    label, conf = runtime.engine.predict(x, weights=runtime.engine.edge_w, edge=True)
    indicators = ["scam-in-progress signal"] if conf >= 0.5 else []
    return {"label": label, "confidence": round(conf, 3), "indicators": indicators}


# ---- dev controls (drive the federation loop / campaign) ------------------

@app.post("/federate/step")
def federate_step() -> dict:
    return runtime.federate_once()


@app.post("/campaign/inject")
def campaign_inject(seeing: bool = True) -> dict:
    # seeing=true: campaign in local training + eval (bank observes the typology).
    # seeing=false: campaign in eval only (bank targeted but blind siloed; the
    # federated model must carry the signal in from other banks).
    runtime.engine.inject_campaign(seeing=seeing)
    _publish({"type": "round_complete", "data": runtime.engine.state()})
    return {"ok": True, "seeing": seeing}
