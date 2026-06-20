import asyncio, json, os, threading, numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from veritas_core.engine import Engine
from veritas_core.model import predict_proba
from veritas_core.gnn_benchmark import compute_gnn_benchmark, benchmark_current

app=FastAPI(title="Veritas Core")
# CORS allowlist is env-driven (comma-separated VERITAS_CORS_ORIGINS) instead of
# a blanket "*", so a deployment only exposes the plane to its own web origin.
# Defaults to common local dev origins for the working demo.
_origins=[o.strip() for o in os.environ.get(
    "VERITAS_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()]
app.add_middleware(CORSMiddleware,allow_origins=_origins,
                   allow_methods=["GET","POST"],allow_headers=["*"])

# Optional shared-secret guard for MUTATING/admin endpoints. If VERITAS_ADMIN_TOKEN
# is set, those endpoints require a matching X-Admin-Token header; if unset (local
# demo), they stay open so the existing UI keeps working.
_ADMIN_TOKEN=os.environ.get("VERITAS_ADMIN_TOKEN")
def _require_admin(token):
    if _ADMIN_TOKEN and token!=_ADMIN_TOKEN:
        raise HTTPException(status_code=401,detail="invalid or missing admin token")

# Human-readable name per model feature index (see data.FEATURE_DIM ordering),
# used to turn real per-feature contributions into explanations.
FEATURE_LABELS=["unusually large amount","origin balance before","origin balance after",
                "destination balance before","destination balance after",
                "new / young account","high-velocity activity",
                "fan-out to multiple recipients","transfer channel",
                "matches active campaign signature"]

engine=Engine(n_banks=8,seed=42); _subs=[]

# Real federated-GNN mule-graph benchmark. It takes a few seconds to TRAIN, so
# compute it ONCE in a background thread at startup; until ready /state omits
# gnnBenchmark (the web shows "GNN pending"). The live `current` block is built
# per request from the REAL round trajectory, keyed on the engine's round.
_gnn_benchmark=None
def _compute_gnn():
    global _gnn_benchmark
    _gnn_benchmark=compute_gnn_benchmark(seed=0)
threading.Thread(target=_compute_gnn,daemon=True).start()

def _state_with_gnn():
    s=engine.state()
    if _gnn_benchmark is not None:
        b=dict(_gnn_benchmark)
        b["current"]=benchmark_current(_gnn_benchmark,engine.round,engine.campaign)
        s["gnnBenchmark"]=b
    return s

def _pub(events):
    for ev in events:
        for q in _subs: q.put_nowait(ev)
@app.get("/state")
def state(): return _state_with_gnn()
@app.get("/banks")
def banks(): return engine.state()["banks"]
@app.get("/provenance")
def provenance(): return engine.provenance
@app.post("/campaign/inject")
def campaign(body:dict,x_admin_token:str=Header(default=None)):
    _require_admin(x_admin_token)
    engine.inject_campaign(); _pub([{"type":"round_complete","data":_state_with_gnn()}]); return {"ok":True}
@app.post("/attack/inject")
def attack(body:dict,x_admin_token:str=Header(default=None)):
    _require_admin(x_admin_token)
    engine.inject_attack(body.get("memberId","bank0")); return {"ok":True}
@app.post("/round/step")
def step(x_admin_token:str=Header(default=None)):
    _require_admin(x_admin_token)
    evs=engine.step()
    for ev in evs:
        if ev.get("type")=="round_complete": ev["data"]=_state_with_gnn()
    _pub(evs); return _state_with_gnn()
@app.post("/sim/reset")
def reset(x_admin_token:str=Header(default=None)):
    _require_admin(x_admin_token)
    global engine; engine=Engine(n_banks=8,seed=42); return engine.state()

def _model_dim():
    # SPEC-DIM: derive the served feature width from the live model object
    # (global weight vector length minus the bias term), NOT a hardcoded literal,
    # so the plane follows the model if its dimension ever changes.
    return len(engine.global_w)-1

def _features_from_tx(tx):
    """Build the model feature vector from the ACTUAL request transaction.

    Reads any provided feature by its FEATURE_LABELS-aligned key (and a few
    friendly aliases); unspecified features default to 0 (neutral). Falls back to
    the 'safe-account mule' campaign signature only when the caller gives no
    usable fields, so the demo's default request still flags as fraud."""
    d=_model_dim()
    x=np.zeros(d)
    # alias map: request key -> feature index
    aliases={"amount":0,"oldOrig":1,"newOrig":2,"oldDest":3,"newDest":4,
             "accountAge":5,"velocity":6,"fanout":7,"isTransfer":8,
             "campaignSignature":d-1,"campaignSig":d-1}
    used=False
    for k,v in (tx or {}).items():
        idx=aliases.get(k)
        if idx is not None and idx<d:
            try: x[idx]=float(v); used=True
            except (TypeError,ValueError): pass
    if not used:
        # default mule signature: benign on generic axes, distinctive on campaign
        if d>=10:
            x[0]=-0.5; x[6]=-0.5; x[7]=-0.5; x[5]=-2.0
        x[d-1]=1.5
    return x

@app.post("/predict")
def predict(body:dict):
    tx=body.get("transaction",{}) or {}
    x=_features_from_tx(tx).reshape(1,-1)
    p=float(predict_proba(engine.global_w,x)[0])
    # Real per-feature contributions: x_i * w_i (logit space). The features that
    # push the score UP the most are the genuine indicators for THIS transaction.
    w=engine.global_w[:x.shape[1]]
    contrib=(x[0]*w)
    order=np.argsort(contrib)[::-1]
    indicators=[FEATURE_LABELS[i] for i in order
                if contrib[i]>1e-9 and i<len(FEATURE_LABELS)][:3]
    if not indicators:
        indicators=["no strong fraud indicators"]
    regime="federated"
    return {"label":"fraud" if p>=0.5 else "legitimate","confidence":round(p,3),
            "score":round(p,3),"regime":regime,"indicators":indicators}
@app.get("/events")
async def events():
    q=asyncio.Queue(); _subs.append(q)
    async def gen():
        try:
            yield {"event":"round_complete","data":json.dumps(_state_with_gnn())}
            while True:
                ev=await q.get(); yield {"event":ev["type"],"data":json.dumps(ev["data"])}
        finally: _subs.remove(q)
    return EventSourceResponse(gen())
