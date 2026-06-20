import { gnnBenchmark } from "./gnnBenchmark.js";

const BANKS=[["bank0","Barclays",2100000],["bank1","NatWest",1900000],
["bank2","Lloyds",1750000],["bank3","HSBC",1600000],["bank4","Santander",1400000],
["bank5","Monzo",900000],["bank6","Starling",700000],["bank7","Nationwide",1500000]];
function detection(regime,round,i,campaign){
  if(!campaign) return 0.9;
  if(regime==="federated") return Math.min(0.97,0.4+0.22*round);
  return Math.min(0.9,Math.max(0.1,0.1+0.12*(round-i*1.2)));
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
