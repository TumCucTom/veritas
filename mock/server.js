import express from "express"; import cors from "cors";
import { snapshot, BANK_IDS } from "./scenario.js";
const app=express(); app.use(cors()); app.use(express.json());
let s={round:0,campaign:false,attack:false};
const nodeClients=new Set(), planeClients=new Set(); const snap=()=>snapshot(s.round,s.campaign,s.attack);
const provenance=()=>Array.from({length:s.round},(_,i)=>({
  round:i+1,
  contributors:BANK_IDS,
  rejected:s.attack&&i===0?[BANK_IDS[0]]:[],
  globalRecall:Math.min(0.97,0.4+0.22*(i+1))
}));
const tenantState=()=>{
  const x=snap(),fed=x.counters.federated,silo=x.counters.siloed;
  return {
    round:x.round,modelVersion:Math.max(1,x.round),customerRecordsTransmitted:0,
    node:{status:"synced",lastSync:new Date().toISOString(),attestation:"verified"},
    lift:{
      federated:{recall:x.banks.reduce((a,b)=>a+b.detection.federated,0)/x.banks.length,
        victims:fed.victims,lostGbp:fed.lostGbp},
      siloed:{recall:x.banks.reduce((a,b)=>a+b.detection.siloed,0)/x.banks.length,
        victims:silo.victims,lostGbp:silo.lostGbp},
      fraudPreventedGbp:fed.fraudPreventedGbp
    },
    edgeFleet:{
      devicesEnrolled:1840000,onLatestModelPct:0.82,modelVersion:Math.max(1,x.round),
      scamInProgressAlerts24h:x.campaignActive?384:112,
      appVersions:[{version:"5.4.x",share:0.61},{version:"5.3.x",share:0.27},{version:"≤5.2",share:0.12}]
    }
  };
};
const registry=()=>[
  {version:Math.max(1,s.round),status:"promoted",metrics:{recall:Math.min(0.97,0.4+0.22*s.round)},createdAt:new Date().toISOString()},
  {version:Math.max(2,s.round+1),status:"canary",metrics:{recall:Math.min(0.97,0.52+0.18*s.round)},createdAt:new Date().toISOString()}
];
const root=()=>({size:Math.max(1,s.round),rootHash:`mock-root-${String(s.round).padStart(4,"0")}-8bb7c0ffee`,
  signaturePem:"-----BEGIN MOCK SIGNATURE-----\\nveritas-demo\\n-----END MOCK SIGNATURE-----"});
const writeEvent=(clients,t,d)=>{for(const r of clients) r.write(`event: ${t}\ndata: ${JSON.stringify(d)}\n\n`);};
const sendNode=(t,d)=>setImmediate(()=>writeEvent(nodeClients,t,d));
const sendPlane=(t,d)=>setImmediate(()=>writeEvent(planeClients,t,d));
app.get("/state",(_q,r)=>r.json(snap()));
app.get("/banks",(_q,r)=>r.json(snap().banks));
app.get("/provenance",(_q,r)=>r.json(provenance()));
app.get("/v1/tenants/:tid/state",(_q,r)=>r.json(tenantState()));
app.get("/v1/tenants/:tid/provenance",(_q,r)=>r.json(provenance()));
app.get("/v1/models/registry",(_q,r)=>r.json(registry()));
app.get("/v1/transparency/root",(_q,r)=>r.json(root()));
app.get("/v1/transparency",(_q,r)=>r.json(provenance().map((p,i)=>({
  seq:i+1,type:"round_aggregated",round:p.round,data:p,leafHash:`leaf-${p.round}`,timestamp:new Date().toISOString()
}))));
app.post("/predict",(_q,r)=>r.json({label:"fraud",confidence:0.97,
  indicators:["new account","high-velocity fan-out to multiple recipients"]}));
app.post("/campaign/inject",(_q,r)=>{s.campaign=true; const x=snap(); r.json({ok:true}); sendNode("round_complete",x); sendPlane("round_complete",tenantState());});
app.post("/attack/inject",(_q,r)=>{s.attack=true; r.json({ok:true}); sendNode("attack_detected",{bankId:BANK_IDS[0],rejected:true}); sendPlane("attack_detected",{bankId:BANK_IDS[0],rejected:true});});
app.post("/sim/reset",(_q,r)=>{s={round:0,campaign:false,attack:false}; r.json(snap());});
app.post("/round/step",(_q,r)=>{s.round++; const x=snap(); r.json(x); sendNode("round_complete",x); sendPlane("round_complete",tenantState());
  for(const b of x.banks) sendNode("client_updated",{bankId:b.id,detection:b.detection});});
app.get("/events",(q,r)=>{r.set({"Content-Type":"text/event-stream","Cache-Control":"no-cache",Connection:"keep-alive"});
  r.flushHeaders(); r.write(`event: round_complete\ndata: ${JSON.stringify(snap())}\n\n`);
  nodeClients.add(r); q.on("close",()=>nodeClients.delete(r));});
app.get("/v1/events",(q,r)=>{r.set({"Content-Type":"text/event-stream","Cache-Control":"no-cache",Connection:"keep-alive"});
  r.flushHeaders(); r.write(`event: round_complete\ndata: ${JSON.stringify(tenantState())}\n\n`);
  planeClients.add(r); q.on("close",()=>planeClients.delete(r));});
app.listen(8001,()=>console.log("Veritas mock :8001"));
