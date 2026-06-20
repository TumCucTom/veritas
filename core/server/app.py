import asyncio, json, threading, numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from veritas_core.engine import Engine
from veritas_core.model import predict_proba
from veritas_core.data import FEATURE_DIM
from veritas_core.gnn_benchmark import compute_gnn_benchmark, benchmark_current

app=FastAPI(title="Veritas Core")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
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
def campaign(body:dict): engine.inject_campaign(); _pub([{"type":"round_complete","data":_state_with_gnn()}]); return {"ok":True}
@app.post("/attack/inject")
def attack(body:dict): engine.inject_attack(body.get("memberId","bank0")); return {"ok":True}
@app.post("/round/step")
def step():
    evs=engine.step()
    for ev in evs:
        if ev.get("type")=="round_complete": ev["data"]=_state_with_gnn()
    _pub(evs); return _state_with_gnn()
@app.post("/sim/reset")
def reset():
    global engine; engine=Engine(n_banks=8,seed=42); return engine.state()
@app.post("/predict")
def predict(body:dict):
    sig=float(body.get("transaction",{}).get("campaignSignature",1.0))
    # mirror the "safe-account mule" signature the engine trains against:
    # benign on generic fraud axes, distinctive on the campaign feature.
    x=np.zeros((1,FEATURE_DIM)); x[0,-1]=1.5*sig; x[0,0]=-0.5; x[0,6]=-0.5; x[0,7]=-0.5; x[0,5]=-2.0
    p=float(predict_proba(engine.global_w,x)[0])
    return {"label":"fraud" if p>=0.5 else "legitimate","confidence":round(p,3),
            "indicators":["new account","high-velocity fan-out to multiple recipients","matches active campaign signature"]}
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
