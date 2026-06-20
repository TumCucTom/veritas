export const LOGICAL_CUSTOMERS = 1_000_000;
export const VISIBLE_POINTS = 32_000;
export const BANK_COMMUNITIES = 8;

export type ContagionRegime = "siloed" | "federated";
export type ContagionState = "neutral" | "exposed" | "protected";

export interface ContagionInput {
  regime: ContagionRegime;
  round: number;
  campaignActive: boolean;
  detection: number;
}

export interface ContagionPoint {
  x: number;
  y: number;
  bank: number;
  state: ContagionState;
  intensity: number;
  hub: boolean;
}

export interface BankBand {
  bank: number;
  cx: number;
  cy: number;
  radius: number;
}

export interface ContagionFrame {
  logicalCustomers: number;
  scalePerPoint: number;
  points: ContagionPoint[];
  bands: BankBand[];
  frontier: {
    fraudRadius: number;
    protectionRadius: number;
  };
  totals: {
    exposed: number;
    protected: number;
    neutral: number;
    suppressed: number;
    crossBankLinks: number;
  };
}

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

const BANK_CENTERS: Array<[number, number]> = [
  [0.16, 0.32],
  [0.35, 0.2],
  [0.57, 0.24],
  [0.78, 0.34],
  [0.23, 0.69],
  [0.43, 0.78],
  [0.64, 0.73],
  [0.84, 0.64],
];

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function seededUnit(index: number, salt: number): number {
  let x = Math.imul(index + 0x9e3779b9, 0x85ebca6b) ^ salt;
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return (x >>> 0) / 0xffffffff;
}

function distance(aX: number, aY: number, bX: number, bY: number): number {
  const dx = aX - bX;
  const dy = aY - bY;
  return Math.sqrt(dx * dx + dy * dy);
}

function nearestCorridorDistance(x: number, y: number, bank: number): number {
  const next = (bank + 3) % BANK_COMMUNITIES;
  const prev = (bank + BANK_COMMUNITIES - 2) % BANK_COMMUNITIES;
  const [ax, ay] = BANK_CENTERS[bank];
  const [bx, by] = BANK_CENTERS[next];
  const [cx, cy] = BANK_CENTERS[prev];
  return Math.min(pointLineDistance(x, y, ax, ay, bx, by), pointLineDistance(x, y, ax, ay, cx, cy));
}

function pointLineDistance(
  x: number,
  y: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): number {
  const vx = bx - ax;
  const vy = by - ay;
  const wx = x - ax;
  const wy = y - ay;
  const lenSq = vx * vx + vy * vy || 1;
  const t = Math.max(0, Math.min(1, (wx * vx + wy * vy) / lenSq));
  const px = ax + t * vx;
  const py = ay + t * vy;
  return distance(x, y, px, py);
}

function frontiers(input: ContagionInput): ContagionFrame["frontier"] {
  if (!input.campaignActive) return { fraudRadius: 0, protectionRadius: 0 };

  const round = Math.max(0, input.round);
  const detection = clamp01(input.detection);
  const fraudBase = input.regime === "siloed" ? 0.18 : 0.12;
  const fraudSpeed = input.regime === "siloed" ? 0.115 : 0.075;
  const fraudResistance = input.regime === "federated" ? detection * 0.32 : detection * 0.1;
  const fraudRadius = clamp01(fraudBase + round * fraudSpeed - fraudResistance);

  const protectionBase = input.regime === "federated" ? 0.12 : 0.04;
  const protectionSpeed = input.regime === "federated" ? 0.14 : 0.04;
  const protectionCap = input.regime === "federated" ? 1 : 0.58;
  const protectionRadius = Math.min(
    protectionCap,
    clamp01((protectionBase + round * protectionSpeed) * (0.35 + detection * 0.9)),
  );

  return { fraudRadius, protectionRadius };
}

function pointForIndex(index: number, input: ContagionInput): ContagionPoint {
  const bank = index % BANK_COMMUNITIES;
  const [cx, cy] = BANK_CENTERS[bank];
  const ring = Math.sqrt(seededUnit(index, 17));
  const angle = index * GOLDEN_ANGLE + seededUnit(index, 41) * 0.8;
  const communityRadius = 0.115 + seededUnit(index, 61) * 0.035;
  const hub = index % 197 === 0 || seededUnit(index, 89) > 0.996;
  const corridorBias = hub ? 0.12 : 0;
  const rawX = cx + Math.cos(angle) * ring * communityRadius + corridorBias * (0.5 - cx);
  const rawY = cy + Math.sin(angle) * ring * communityRadius + corridorBias * (0.5 - cy);
  const x = Math.max(0.025, Math.min(0.975, rawX));
  const y = Math.max(0.055, Math.min(0.945, rawY));

  const { fraudRadius, protectionRadius } = frontiers(input);
  const sourceDistance = Math.min(distance(x, y, 0.14, 0.2), distance(x, y, 0.83, 0.78));
  const communityDistance = distance(x, y, cx, cy);
  const corridorDistance = nearestCorridorDistance(x, y, bank);
  const hubBoost = hub ? 0.2 : 0;
  const spreadScore = clamp01(
    sourceDistance * 0.92 + communityDistance * 1.15 + corridorDistance * 0.7 - hubBoost,
  );
  const protectionScore = clamp01(
    distance(x, y, 0.5, 0.5) * 0.78 + communityDistance * 0.48 - (hub ? 0.06 : 0),
  );

  const fraudNoise = seededUnit(index, 131) * 0.13;
  const protectionNoise = seededUnit(index, 197) * 0.11;
  const protectedByWave = input.campaignActive && protectionScore + protectionNoise < protectionRadius;
  const exposedByWave = input.campaignActive && spreadScore + fraudNoise < fraudRadius;

  const state: ContagionState = protectedByWave ? "protected" : exposedByWave ? "exposed" : "neutral";
  const intensity =
    state === "protected"
      ? clamp01(1 - protectionScore + protectionRadius * 0.4)
      : state === "exposed"
        ? clamp01(1 - spreadScore + fraudRadius * 0.35)
        : 0.35 + seededUnit(index, 233) * 0.3;

  return { x, y, bank, state, intensity, hub };
}

function computeTotals(points: ContagionPoint[]): ContagionFrame["totals"] {
  let exposedVisible = 0;
  let protectedVisible = 0;
  let hubLinks = 0;
  for (const point of points) {
    if (point.state === "exposed") exposedVisible++;
    if (point.state === "protected") protectedVisible++;
    if (point.hub) hubLinks++;
  }

  const scale = LOGICAL_CUSTOMERS / VISIBLE_POINTS;
  const exposed = Math.round(exposedVisible * scale);
  const protectedCount = Math.round(protectedVisible * scale);
  const neutral = Math.max(0, LOGICAL_CUSTOMERS - exposed - protectedCount);
  const suppressed = Math.max(0, protectedCount - Math.round(exposed * 0.24));

  return {
    exposed,
    protected: protectedCount,
    neutral,
    suppressed,
    crossBankLinks: Math.round(hubLinks * scale * 0.18),
  };
}

export function computeContagionFrame(input: ContagionInput): ContagionFrame {
  const points: ContagionPoint[] = [];
  for (let i = 0; i < VISIBLE_POINTS; i++) {
    points.push(pointForIndex(i, input));
  }

  return {
    logicalCustomers: LOGICAL_CUSTOMERS,
    scalePerPoint: LOGICAL_CUSTOMERS / VISIBLE_POINTS,
    points,
    bands: BANK_CENTERS.map(([cx, cy], bank) => ({
      bank,
      cx,
      cy,
      radius: 0.14,
    })),
    frontier: frontiers(input),
    totals: computeTotals(points),
  };
}
