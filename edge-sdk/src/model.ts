/**
 * On-device logistic model — a faithful TypeScript port of
 * core/veritas_core/model.py. Same shape: sigmoid(aug(X) @ w), trained with
 * full-batch gradient steps, dim = FEATURE_DIM + 1 (the trailing bias term).
 *
 * Pure TS, no native deps, so it runs in a browser or on a phone runtime.
 */

/** Feature contract shared with the Python core (data.py FEATURE_DIM). */
export const FEATURE_DIM = 10;

/** Model dimension including the logistic bias term. */
export const MODEL_DIM = FEATURE_DIM + 1;

export type Vector = number[];
export type Matrix = number[][];

/** init_weights(dim) -> zeros(dim + 1). */
export function initWeights(dim: number = FEATURE_DIM): Vector {
  return new Array<number>(dim + 1).fill(0);
}

/** Numerically-stable sigmoid with the same clip(-30, 30) as the core. */
function sigmoid(z: number): number {
  const c = z < -30 ? -30 : z > 30 ? 30 : z;
  return 1.0 / (1.0 + Math.exp(-c));
}

/** Dot product of an augmented feature row [x..., 1] with weights w. */
function dotAug(row: Vector, w: Vector): number {
  // w has length dim + 1; the final weight is the bias (× implicit 1).
  let acc = w[row.length] ?? 0;
  for (let j = 0; j < row.length; j++) acc += row[j]! * w[j]!;
  return acc;
}

/** predict_proba(w, X) over a batch. */
export function predictProba(w: Vector, X: Matrix): Vector {
  return X.map((row) => sigmoid(dotAug(row, w)));
}

/** Single-row probability (on-device scoring path). */
export function predictProbaOne(w: Vector, x: Vector): number {
  return sigmoid(dotAug(x, w));
}

export interface TrainOpts {
  epochs?: number;
  lr?: number;
  l2?: number;
}

/**
 * train_local(w, X, y) — full-batch logistic gradient descent with class
 * re-weighting for the (rare) positive class, mirroring model.py exactly:
 *
 *   pw = (#neg / #pos) for positives, else 1
 *   g  = Xa^T @ ((sigmoid(Xa @ w) - y) * pw) / n + l2 * w
 *   w -= lr * g
 */
export function trainLocal(
  w0: Vector,
  X: Matrix,
  y: Vector,
  opts: TrainOpts = {},
): Vector {
  const { epochs = 10, lr = 0.3, l2 = 1e-4 } = opts;
  const n = y.length;
  if (n === 0) return w0.slice();
  const dim = w0.length; // FEATURE_DIM + 1
  const feat = dim - 1;

  const npos = y.reduce((s, v) => s + (v === 1 ? 1 : 0), 0);
  const nneg = n - npos;
  const posWeight = nneg / Math.max(1, npos);
  const pw = y.map((v) => (v === 1 ? posWeight : 1.0));

  const w = w0.slice();
  for (let e = 0; e < epochs; e++) {
    const grad = new Array<number>(dim).fill(0);
    for (let i = 0; i < n; i++) {
      const row = X[i]!;
      const err = (sigmoid(dotAug(row, w)) - y[i]!) * pw[i]!;
      for (let j = 0; j < feat; j++) grad[j]! += row[j]! * err;
      grad[feat]! += err; // bias column is implicit 1
    }
    for (let j = 0; j < dim; j++) {
      grad[j]! = grad[j]! / n + l2 * w[j]!;
      w[j]! -= lr * grad[j]!;
    }
  }
  return w;
}

/** recall(w, X, y, thr) — same definition as the core. */
export function recall(w: Vector, X: Matrix, y: Vector, thr = 0.5): number {
  const totalPos = y.reduce((s, v) => s + (v === 1 ? 1 : 0), 0);
  if (totalPos === 0) return 1.0;
  const probs = predictProba(w, X);
  let tp = 0;
  for (let i = 0; i < y.length; i++) {
    if (probs[i]! > thr && y[i] === 1) tp++;
  }
  return tp / totalPos;
}

/** Elementwise weight delta (new - base). */
export function subtract(a: Vector, b: Vector): Vector {
  return a.map((v, i) => v - (b[i] ?? 0));
}

/** L2 norm of a vector. */
export function l2norm(u: Vector): number {
  let s = 0;
  for (const v of u) s += v * v;
  return Math.sqrt(s);
}
