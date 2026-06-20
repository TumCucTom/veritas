import { gnnBenchmark } from "./gnnBenchmark.js";

const BANKS=[["bank0","Barclays",2100000],["bank1","NatWest",1900000],
["bank2","Lloyds",1750000],["bank3","HSBC",1600000],["bank4","Santander",1400000],
["bank5","Monzo",900000],["bank6","Starling",700000],["bank7","Nationwide",1500000]];
// Per-bank "personality" so the eight institutions never read identically.
// FED_CAP: where federation lifts each bank to (all high — shared intelligence).
// SILO_CAP: where each bank plateaus ALONE (low — so the siloed map stays red).
const FED_CAP =[0.972,0.963,0.958,0.969,0.961,0.949,0.955,0.967];
const SILO_CAP=[0.66,0.61,0.58,0.64,0.55,0.49,0.52,0.69];
function detection(regime,round,i,campaign){
  if(!campaign) return Math.min(FED_CAP[i],0.9); // calm baseline, slight per-bank spread
  if(regime==="federated")
    // Federation pulls every bank quickly toward its (high) ceiling.
    return Math.max(0.4,Math.min(FED_CAP[i],0.5+0.17*round+0.004*i));
  // Siloed: each bank is hit as the campaign propagates (onset ~ i), crawls up
  // slowly on its own data, and plateaus LOW — the visible red contagion.
  const onset=i*1.1, t=Math.max(0,round-onset);
  return Math.max(0.12,Math.min(SILO_CAP[i],0.16+0.055*t));
}
export function snapshot(round,campaign,attack){
  const banks=BANKS.map(([id,name,customers],i)=>({id,name,customers,
    poisoned: attack&&id==="bank0"&&round<2,
    detection:{federated:detection("federated",round,i,campaign),
               siloed:detection("siloed",round,i,campaign)}}));
  const fd=banks.reduce((s,b)=>s+b.detection.federated,0)/banks.length;
  const sd=banks.reduce((s,b)=>s+b.detection.siloed,0)/banks.length;
  const AR=9000,AVG=255,fv=Math.round(AR*(1-fd)),sv=Math.round(AR*(1-sd));
  return {round,running:true,banks,campaignActive:campaign,attackActive:attack,
    customerRecordsTransmitted:0,
    gnnBenchmark:gnnBenchmark(round,campaign),
    counters:{federated:{fraudPreventedGbp:(sv-fv)*AVG,timeToDetectHours:3,victims:fv,lostGbp:fv*AVG},
              siloed:{fraudPreventedGbp:0,timeToDetectHours:101,victims:sv,lostGbp:sv*AVG}}};
}
export const BANK_IDS=BANKS.map(b=>b[0]);
