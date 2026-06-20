# Veritas API contract (v1)
Base URL: `http://localhost:8000`  (mock: `http://localhost:8001`)

## REST
GET  /state    -> State (snapshot below)
GET  /banks    -> Bank[]
GET  /provenance -> Provenance[]                (per-round model bill-of-materials)
POST /predict  {transaction:{...}, text?} -> {label, confidence, indicators[]}
POST /campaign/inject {typology} -> {ok:true}   (starts fraud campaign, both regimes)
POST /attack/inject   {memberId} -> {ok:true}   (member submits poisoned updates)
POST /round/step      -> State                  (advance one federated round)
POST /sim/reset       -> State

## SSE  GET /events (text/event-stream), named events:
round_complete  -> State
client_updated  -> {bankId, detection:{federated,siloed}}
fraud_propagated-> {bankId, regime}
attack_detected -> {bankId, rejected}

## State
{ round:int, running:bool, banks:Bank[], counters:Counters,
  campaignActive:bool, attackActive:bool, customerRecordsTransmitted:0 }
Bank = { id, name, customers:int, detection:{federated:float,siloed:float}, poisoned:bool }
Counters = { federated:Regime, siloed:Regime }
Regime = { fraudPreventedGbp, timeToDetectHours, victims, lostGbp }
Provenance = { round:int, contributors:string[], rejected:string[], globalRecall:float }
  (per federated round: member banks FedAvg'd in vs the dropped poisoned update;
   production path anchors each record on FLock on-chain attestation)
