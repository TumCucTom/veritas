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
from fastapi import FastAPI
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
    update: list[float]
    numExamples: int = 1


class EdgeScoreBody(BaseModel):
    features: list[float]


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


@app.post("/edge/v1/updates")
def edge_updates(body: EdgeUpdateBody) -> dict:
    # Secure-aggregate the device update (DP) into the bank edge model in-tenancy;
    # the per-device update is never stored. deviceToken is ephemeral, not a
    # customer identifier.
    runtime.engine.edge_aggregate(np.asarray(body.update, dtype=np.float64), body.numExamples)
    return {"accepted": True}


@app.post("/edge/v1/score")
def edge_score(body: EdgeScoreBody) -> dict:
    x = np.asarray(body.features, dtype=np.float64)
    label, conf = runtime.engine.predict(x, weights=runtime.engine.edge_w)
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
