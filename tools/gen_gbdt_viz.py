"""OFFLINE generator for the Federated GBDT visualization (web/public/viz/gbdt.json).

WHAT THIS SHOWS
---------------
Veritas trains histogram-based gradient-boosted trees ACROSS banks. The only
thing that ever leaves a bank is a per-feature HISTOGRAM (bin counts / gradient
sums) — never a raw customer row. The coordinator SECURELY SUMS the per-bank
histograms (Bonawitz pairwise masking, ``secure_agg.secure_sum``) into one
global histogram, and the split for each tree node is chosen from that GLOBAL
histogram alone.

This script exercises the REAL core:
  * ``veritas_core.data.make_bank_data``  — per-bank (X, y) shards. Feature 0
    ("amount") is a genuine fraud axis (fraud rows are shifted +1.5), so it is a
    representative feature whose histogram visibly separates the classes.
  * ``veritas_core.fedgbdt._coarse_count_hist`` — the SAME additive per-feature
    count histogram the trainer uses to derive bin edges. We compute one per
    bank for the representative feature.
  * ``veritas_core.fedgbdt.secure_sum`` — the secure aggregation primitive.
    global histogram == secure_sum(per-bank histograms), bit-exactly.
  * ``veritas_core.fedgbdt.fed_train_gbdt`` — the real federated GBDT. We read
    the FIRST tree's root split (feature, threshold-in-bin-space, gain) and a
    few of its nodes back out for the resulting-tree panel, and report recall.

Everything is deterministic (fixed seed) and the JSON is < 100 KB.

Run:
    source /Users/tom/Veritas/core/.venv/bin/activate
    python tools/gen_gbdt_viz.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# Make veritas_core importable when run from repo root.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "core"))

from veritas_core import data  # noqa: E402
from veritas_core.fedgbdt import (  # noqa: E402
    _coarse_count_hist,
    _best_split,
    _bin_features,
    _build_edges,
    _node_histograms,
    _grad_hess,
    _sigmoid,
    secure_sum,
    fed_train_gbdt,
    predict_proba,
    recall,
)

SEED = 1234
N_BANKS = 4
N_PER_BANK = 1500
FRAUD_RATE = 0.18
N_BINS_VIZ = 12  # histogram bars shown per bank / global (keeps JSON tiny)
GRID = 256  # coarse grid the real edge-derivation uses
FEATURE_IDX = 0  # "amount" — a genuine fraud axis (fraud rows shifted +1.5)
FEATURE_NAMES = [
    "amount", "oldOrig", "newOrig", "oldDest", "newDest",
    "accountAge", "velocity", "fanout", "isTransfer", "campaignSig",
]

OUT_DIR = os.path.join(REPO, "web", "public", "viz")
OUT_PATH = os.path.join(OUT_DIR, "gbdt.json")


def main() -> None:
    rng = np.random.default_rng(SEED)

    # --- per-bank shards (raw rows that NEVER leave a bank) ------------------
    bank_data = []
    for b in range(N_BANKS):
        X, y = data.make_bank_data(N_PER_BANK, FRAUD_RATE, seed=SEED + b)
        bank_data.append((X, y))
    bank_X = [X for X, _ in bank_data]

    # --- shared coarse grid for the representative feature ------------------
    # Reuse the SAME [lo, hi] grid the real edge-derivation uses so the bin
    # labels are meaningful and additive. Then coarsen GRID -> N_BINS_VIZ.
    lo = np.min([X.min(axis=0) for X in bank_X], axis=0).astype(np.float64)
    hi = np.max([X.max(axis=0) for X in bank_X], axis=0).astype(np.float64)
    span = hi - lo
    span[span == 0] = 1.0
    hi = lo + span
    grid_centers = lo[:, None] + (np.arange(GRID) + 0.5) * (span[:, None] / GRID)

    # Per-bank coarse (n_features x GRID) count histograms (the REAL primitive).
    coarse = [_coarse_count_hist(X, lo, hi, GRID) for X in bank_X]

    # Coarsen GRID columns into N_BINS_VIZ display bins for the representative
    # feature only. ``np.add.reduceat`` keeps counts additive, so the coarsened
    # per-bank histograms still secure-sum to the coarsened global histogram.
    edges_idx = np.linspace(0, GRID, N_BINS_VIZ + 1).astype(int)
    starts = edges_idx[:-1]

    def coarsen(feat_hist_grid: np.ndarray) -> np.ndarray:
        row = feat_hist_grid[FEATURE_IDX]
        return np.add.reduceat(row, starts).astype(np.int64)

    bank_hists = [coarsen(c) for c in coarse]

    # Display bin labels: amount value at the centre of each display bin.
    fc = grid_centers[FEATURE_IDX]
    bin_centers = np.array(
        [fc[(s + e) // 2] for s, e in zip(edges_idx[:-1], edges_idx[1:])]
    )
    bins = [round(float(v), 3) for v in bin_centers]

    # --- GLOBAL histogram == secure_sum(per-bank histograms) ----------------
    # Use the REAL secure aggregation primitive. Masks cancel exactly, so the
    # recovered global histogram is the elementwise sum of the per-bank ones.
    global_hist = secure_sum(
        [h.astype(np.float64) for h in bank_hists], secure=True, session=42
    ).astype(np.int64)

    # Independent verification that the secure sum equals the plain sum.
    plain_sum = np.sum(np.stack(bank_hists), axis=0)
    assert np.array_equal(global_hist, plain_sum), "secure_sum != elementwise sum"

    # --- the REAL chosen split on the representative feature ----------------
    # Train the federated GBDT, then read the FIRST tree's structure. We also
    # recompute the root's gradient/hessian global histogram so we can report
    # the gain on THIS feature exactly as ``_best_split`` evaluates it.
    model = fed_train_gbdt(
        bank_data, n_trees=12, max_depth=3, n_bins=16, lr=0.3, secure=True
    )

    # Recompute the root-node federated histograms to obtain the per-feature
    # best split and gain (mirrors _grow_tree's first node exactly).
    edges = model.edges
    n_bins_per_f = [e.size + 1 for e in edges]
    bank_binned = [_bin_features(X, edges) for X in bank_X]
    base_score = model.base_score
    per_bank_gh = []
    n_features = bank_X[0].shape[1]
    for b in range(N_BANKS):
        rows = np.arange(bank_X[b].shape[0])
        logit = np.full(bank_X[b].shape[0], base_score, dtype=np.float64)
        p = _sigmoid(logit)
        g, h = _grad_hess(p, bank_data[b][1].astype(np.float64))
        per_bank_gh.append(
            _node_histograms(bank_binned[b], g, h, rows, n_features, n_bins_per_f)
        )
    glob_gh = secure_sum(per_bank_gh, secure=True, session=1000)
    Hg, Hh = glob_gh[0], glob_gh[1]

    # Best split chosen by the REAL routine across ALL features (the model's
    # actual root choice), plus the gain restricted to the representative
    # feature for the histogram overlay.
    best = _best_split(Hg, Hh, n_bins_per_f, lam=1.0, gamma=0.0, min_hess=1.0)

    # Map the representative-feature split (in fedgbdt bin space, n_bins=16)
    # onto our N_BINS_VIZ display bins so the vertical line lands sensibly. We
    # also compute the gain of the best split ON the representative feature.
    feat_edges = edges[FEATURE_IDX]
    feat_threshold_value = None
    split_gain = None
    split_display_bin = N_BINS_VIZ // 2

    # gain of the best representative-feature split (single-feature scan)
    G = float(Hg.sum(axis=1)[0])
    H = float(Hh.sum(axis=1)[0])
    base = (G * G) / (H + 1.0)
    nb = n_bins_per_f[FEATURE_IDX]
    gl = np.cumsum(Hg[FEATURE_IDX, :nb])
    hl = np.cumsum(Hh[FEATURE_IDX, :nb])
    best_k_feat, best_gain_feat = None, -np.inf
    for k in range(nb - 1):
        GL, HL = gl[k], hl[k]
        GR, HR = G - GL, H - HL
        if HL < 1.0 or HR < 1.0:
            continue
        gain = 0.5 * ((GL * GL) / (HL + 1.0) + (GR * GR) / (HR + 1.0) - base)
        if gain > best_gain_feat:
            best_gain_feat, best_k_feat = gain, k

    if best_k_feat is not None:
        # bin index k -> threshold value via the feature's edges
        if best_k_feat < feat_edges.size:
            feat_threshold_value = float(feat_edges[best_k_feat])
        else:
            feat_threshold_value = float(feat_edges[-1])
        split_gain = float(best_gain_feat)
        # locate threshold value among display bin centres
        split_display_bin = int(np.searchsorted(bin_centers, feat_threshold_value))
        split_display_bin = int(np.clip(split_display_bin, 0, N_BINS_VIZ - 1))

    # --- resulting small tree (first tree, first few nodes) -----------------
    tree0 = model.trees[0]
    feat_arr = tree0["feature"]
    thr_arr = tree0["threshold"]
    left_arr = tree0["left"]
    right_arr = tree0["right"]

    def threshold_value(node: int) -> float:
        """Convert stored bin-space threshold to an amount value for display."""
        f = int(feat_arr[node])
        if f < 0:
            return 0.0
        k = int(thr_arr[node])
        e = edges[f]
        return float(e[k]) if 0 <= k < e.size else float(e[-1]) if e.size else 0.0

    # parent/side bookkeeping via BFS over child arrays, capped to a few nodes.
    parent = {0: None}
    side = {0: None}
    for nid in range(len(feat_arr)):
        l, r = int(left_arr[nid]), int(right_arr[nid])
        if l >= 0:
            parent[l] = nid
            side[l] = "L"
        if r >= 0:
            parent[r] = nid
            side[r] = "R"

    tree_nodes = []
    MAX_TREE_NODES = 7
    for nid in range(min(len(feat_arr), MAX_TREE_NODES)):
        f = int(feat_arr[nid])
        is_leaf = f < 0
        tree_nodes.append(
            {
                "id": nid,
                "feature": "leaf" if is_leaf else FEATURE_NAMES[f],
                "threshold": 0.0 if is_leaf else round(threshold_value(nid), 3),
                "parent": parent.get(nid),
                "side": side.get(nid),
            }
        )

    # --- model metric (cheap recall on the pooled data) ---------------------
    X_all = np.vstack(bank_X)
    y_all = np.concatenate([y for _, y in bank_data])
    metric = round(float(recall(model, X_all, y_all)), 4)

    out = {
        "feature": FEATURE_NAMES[FEATURE_IDX],
        "bins": bins,
        "bankHistograms": [
            {"bank": b, "counts": [int(c) for c in bank_hists[b]]}
            for b in range(N_BANKS)
        ],
        "globalHistogram": [int(c) for c in global_hist],
        "split": {
            "binIndex": int(split_display_bin),
            "gain": round(float(split_gain if split_gain is not None else 0.0), 4),
        },
        "tree": tree_nodes,
        "meta": {
            "note": (
                "Each bank sends only privacy-masked feature histograms (bin "
                "counts), never raw customer rows. The coordinator securely sums "
                "them (Bonawitz pairwise masking) into one global histogram and "
                "picks the best-gain split from that global view alone."
            ),
            "model": "federated histogram GBDT",
            "metric": metric,
        },
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    # --- validate + report --------------------------------------------------
    with open(OUT_PATH) as f:
        reloaded = json.load(f)
    assert reloaded["globalHistogram"] == out["globalHistogram"]
    size = os.path.getsize(OUT_PATH)

    print(f"wrote {OUT_PATH} ({size} bytes, < 100KB: {size < 100_000})")
    print(f"representative feature: {out['feature']}")
    print(f"display bins ({N_BINS_VIZ}): centres = {bins}")
    print()
    print("per-bank histograms (counts, not records):")
    for b in range(N_BANKS):
        row = bank_hists[b]
        print(f"  bank {b}: sum={int(row.sum()):5d}  {list(int(c) for c in row)}")
    print(f"  PLAIN  elementwise sum : {list(int(c) for c in plain_sum)}")
    print(f"  SECURE global histogram: {list(int(c) for c in global_hist)}")
    print(
        f"  global == sum(banks): "
        f"{np.array_equal(global_hist, plain_sum)} "
        f"(global total={int(global_hist.sum())}, "
        f"banks total={int(sum(int(h.sum()) for h in bank_hists))})"
    )
    print()
    print(
        f"chosen split (real fedgbdt): display binIndex={split_display_bin} "
        f"(amount threshold ~= {feat_threshold_value}), gain={out['split']['gain']}"
    )
    print(f"root tree split: feature={best[0] if best else None} "
          f"({FEATURE_NAMES[best[0]] if best else 'none'}), "
          f"bin={best[1] if best else None}, gain={round(best[2],4) if best else None}")
    print(f"resulting tree nodes ({len(tree_nodes)}):")
    for nd in tree_nodes:
        print(f"  {nd}")
    print(f"model metric (recall): {metric}")
    print("JSON validated OK.")


if __name__ == "__main__":
    main()
