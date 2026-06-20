import express from "express"; import cors from "cors";
import { snapshot, BANK_IDS } from "./scenario.js";
const app=express(); app.use(cors()); app.use(express.json());
let s={round:0,campaign:false,attack:false};
const clients=new Set(); const snap=()=>snapshot(s.round,s.campaign,s.attack);
const send=(t,d)=>{for(const r of clients) r.write(`event: ${t}\ndata: ${JSON.stringify(d)}\n\n`);};
app.get("/state",(_q,r)=>r.json(snap()));
app.get("/banks",(_q,r)=>r.json(snap().banks));
app.post("/predict",(_q,r)=>r.json({label:"fraud",confidence:0.97,
  indicators:["new account","high-velocity fan-out to multiple recipients"]}));
app.post("/campaign/inject",(_q,r)=>{s.campaign=true; send("round_complete",snap()); r.json({ok:true});});
app.post("/attack/inject",(_q,r)=>{s.attack=true; send("attack_detected",{bankId:BANK_IDS[0],rejected:true}); r.json({ok:true});});
app.post("/sim/reset",(_q,r)=>{s={round:0,campaign:false,attack:false}; r.json(snap());});
app.post("/round/step",(_q,r)=>{s.round++; const x=snap(); send("round_complete",x);
  for(const b of x.banks) send("client_updated",{bankId:b.id,detection:b.detection}); r.json(x);});
app.get("/events",(q,r)=>{r.set({"Content-Type":"text/event-stream","Cache-Control":"no-cache",Connection:"keep-alive"});
  r.flushHeaders(); r.write(`event: round_complete\ndata: ${JSON.stringify(snap())}\n\n`);
  clients.add(r); q.on("close",()=>clients.delete(r));});
app.listen(8001,()=>console.log("Veritas mock :8001"));
