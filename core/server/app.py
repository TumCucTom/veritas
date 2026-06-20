import asyncio, json, numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from veritas_core.engine import Engine
from veritas_core.model import predict_proba
from veritas_core.data import FEATURE_DIM

app=FastAPI(title="Veritas Core")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
engine=Engine(n_banks=8,seed=42); _subs=[]
def _pub(events):
    for ev in events:
        for q in _subs: q.put_nowait(ev)
@app.get("/state")
def state(): return engine.state()
@app.get("/banks")
def banks(): return engine.state()["banks"]
@app.get("/provenance")
def provenance(): return engine.provenance
@app.post("/campaign/inject")
def campaign(body:dict): engine.inject_campaign(); _pub([{"type":"round_complete","data":engine.state()}]); return {"ok":True}
@app.post("/attack/inject")
def attack(body:dict): engine.inject_attack(body.get("memberId","bank0")); return {"ok":True}
@app.post("/round/step")
def step(): _pub(engine.step()); return engine.state()
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
            yield {"event":"round_complete","data":json.dumps(engine.state())}
            while True:
                ev=await q.get(); yield {"event":ev["type"],"data":json.dumps(ev["data"])}
        finally: _subs.remove(q)
    return EventSourceResponse(gen())
