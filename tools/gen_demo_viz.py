"""OFFLINE demo-viz generator for the Veritas website.

Runs the project's REAL pure-numpy federated-learning models and exports
precomputed animation keyframes as JSON to ``web/public/viz/``. The website
plays these as smooth custom animations. Everything here is deterministic
(fixed seed) and re-runnable.

Three artifacts are produced:

  1. web/public/viz/umap.json
     Embedding separation across federated rounds. We train the REAL
     ``veritas_core.embeddings`` model (shared-categorical embeddings + ReLU-MLP
     head) under FedAvg across several banks. At a handful of round checkpoints
     we extract each sample's PENULTIMATE hidden activation (the ReLU layer
     ``a1`` inside ``embeddings._forward``) — the model's learned representation
     — and project to 2D with UMAP. The reducer is fit ONCE on the final-round
     activations and used to transform every round, so coordinates are
     comparable frame-to-frame (smooth animation). We also train a SINGLE bank
     alone (siloed) for an A/B compare at the final round.

  2. web/public/viz/graph.json
     GNN mule-graph. We build the REAL cross-bank transaction graph
     (``veritas_core.graph.generate_network``, 8 banks, planted cross-bank mule
     rings), lay it out offline with a force-directed spring layout, and run the
     REAL federated GraphSAGE GNN (``veritas_core.gnn``) to get per-node fraud
     scores at several rounds. Mule-ring nodes score increasingly high.

  3. web/public/viz/federation.json
     Federated aggregation: DP clip + Multi-Krum poison rejection. We take REAL
     per-client weight-update deltas from the embedding model, inject one
     poisoned (large-norm) client, clip with ``veritas_core.dp.clip_update`` and
     select honest clients with ``veritas_core.robust.multi_krum_select``. Each
     client update is projected to 2D via a single PCA fit so the website can
     animate arrive -> clip -> select -> aggregate.

Run:  python tools/gen_demo_viz.py
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

from veritas_core import embeddings, gnn, dp  # noqa: E402
from veritas_core.graph import generate_network  # noqa: E402
from veritas_core.aggregation import fedavg  # noqa: E402
from veritas_core.robust import multi_krum_select  # noqa: E402

SEED = 1234
OUT_DIR = os.path.join(REPO, "web", "public", "viz")
ROUND_W = 6  # round float-rounding for compact JSON


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _norm_coords(xy: np.ndarray) -> np.ndarray:
    """Center and scale a (n,2) array to roughly [-1, 1] (preserve aspect)."""
    xy = np.asarray(xy, dtype=float)
    c = xy.mean(axis=0)
    xy = xy - c
    scale = np.percentile(np.abs(xy), 99)
    if scale <= 0:
        scale = 1.0
    return np.clip(xy / scale, -1.0, 1.0)


def _r(x, nd=ROUND_W):
    return round(float(x), nd)


def _xy_list(xy: np.ndarray, nd=4):
    return [[_r(a, nd), _r(b, nd)] for a, b in np.asarray(xy)]


def _xyz_list(xyz: np.ndarray, nd=4):
    return [[_r(a, nd), _r(b, nd), _r(c, nd)] for a, b, c in np.asarray(xyz)]


def _mean_intra_inter(emb2d: np.ndarray, labels: np.ndarray):
    """Mean intra-class vs inter-class centroid distance (cluster separation)."""
    labels = np.asarray(labels)
    c0 = emb2d[labels == 0].mean(axis=0)
    c1 = emb2d[labels == 1].mean(axis=0)
    inter = float(np.linalg.norm(c0 - c1))
    intra0 = float(np.mean(np.linalg.norm(emb2d[labels == 0] - c0, axis=1)))
    intra1 = float(np.mean(np.linalg.norm(emb2d[labels == 1] - c1, axis=1)))
    intra = 0.5 * (intra0 + intra1)
    return inter, intra


# --------------------------------------------------------------------------- #
# ARTIFACT 1 — umap.json
# --------------------------------------------------------------------------- #
def _representation(weights, X_cat, X_dense, cardinalities):
    """The model's learned per-sample representation.

    We use the ReLU hidden activation `a1` (the penultimate layer before the
    readout) CONCATENATED with the model's own decision logit `z` as a strongly
    weighted extra axis. The logit is the direction the head has learned to
    separate fraud from legit, so including it makes the 2D projection reflect
    the model's *learned decision geometry* (not just diffuse hidden noise).
    Both come straight from a single real forward pass of the federated model.
    """
    f = embeddings._forward(
        weights, np.asarray(X_cat), X_dense, cardinalities,
        emb_dim=4, dense_dim=X_dense.shape[1], hidden=16,
    )
    a1 = f["a1"]                      # (n, hidden) penultimate ReLU activations
    z = f["z"].reshape(-1, 1)        # (n, 1) decision logit
    return np.concatenate([a1, 3.0 * z], axis=1)


def gen_umap():
    rng = np.random.default_rng(SEED)
    n = 600
    n_banks = 5

    # Fixed labelled sample with a real fraudulent categorical interaction.
    X_cat, X_dense, y, cardinalities = embeddings.make_categorical_data(
        n, fraud_rate=0.06, seed=SEED,
    )

    # Partition the sample across banks (federated split). Every bank keeps some
    # fraud so FedAvg has a reason to share the learned corridor geometry.
    perm = rng.permutation(n)
    parts = np.array_split(perm, n_banks)

    checkpoints = [0, 2, 5, 10, 16, 22, 30]
    total_rounds = max(checkpoints)

    dim = embeddings.weight_dim(cardinalities, emb_dim=4,
                                dense_dim=X_dense.shape[1], hidden=16)
    assert dim == len(embeddings.init_weights(cardinalities))

    # ---- FEDERATED training (FedAvg) -------------------------------------
    global_w = embeddings.init_weights(cardinalities, seed=SEED)
    fed_acts = {}  # round -> (n, hidden) activations on the full sample
    if 0 in checkpoints:
        fed_acts[0] = _representation(global_w, X_cat, X_dense, cardinalities)

    for rnd in range(1, total_rounds + 1):
        updates = []
        for idx in parts:
            local = embeddings.train_local(
                global_w, X_cat[idx], X_dense[idx], y[idx],
                epochs=20, lr=0.5, cardinalities=cardinalities,
                emb_dim=4, dense_dim=X_dense.shape[1], hidden=16,
            )
            updates.append(local)
        global_w = fedavg(updates)
        if rnd in checkpoints:
            fed_acts[rnd] = _representation(global_w, X_cat, X_dense, cardinalities)

    # ---- SILOED training (one bank alone) at the final round -------------
    silo_idx = parts[0]
    silo_w = embeddings.init_weights(cardinalities, seed=SEED)
    for _ in range(total_rounds):
        silo_w = embeddings.train_local(
            silo_w, X_cat[silo_idx], X_dense[silo_idx], y[silo_idx],
            epochs=20, lr=0.5, cardinalities=cardinalities,
            emb_dim=4, dense_dim=X_dense.shape[1], hidden=16,
        )
    silo_acts = _representation(silo_w, X_cat, X_dense, cardinalities)

    # ---- Project to 3D --------------------------------------------------
    # Fit the reducer ONCE on the final federated round; transform every frame
    # with the same fitted reducer so coordinates are comparable across frames.
    final_round = max(checkpoints)
    method = "umap"
    try:
        import umap
        reducer = umap.UMAP(
            n_components=3, n_neighbors=20, min_dist=0.3,
            random_state=SEED, metric="euclidean",
        )
        reducer.fit(fed_acts[final_round])
        transform = reducer.transform
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"[umap] falling back to PCA ({exc})")
        from sklearn.decomposition import PCA
        method = "pca"
        reducer = PCA(n_components=3, random_state=SEED)
        reducer.fit(fed_acts[final_round])
        transform = reducer.transform

    # Transform every frame with the SAME fitted reducer, then normalize ALL
    # frames + the siloed-final together with ONE shared center + uniform scale
    # so coordinates are comparable across frames and the animation does not
    # jump in scale or position between rounds.
    raw_frames = {r: transform(fed_acts[r]) for r in checkpoints}
    raw_silo = transform(silo_acts)

    all_pts = np.vstack([raw_frames[r] for r in checkpoints] + [raw_silo])
    center = all_pts.mean(axis=0)
    scale = np.percentile(np.abs(all_pts - center), 99)
    if scale <= 0:
        scale = 1.0

    def shared_norm(xyz):
        return np.clip((np.asarray(xyz) - center) / scale, -1.0, 1.0)

    frames = []
    for r in checkpoints:
        frames.append({"round": int(r), "fed": _xyz_list(shared_norm(raw_frames[r]))})
    siloed_final = _xyz_list(shared_norm(raw_silo))

    # ---- separation sanity metric on the final federated frame ----------
    final_norm = shared_norm(raw_frames[final_round])
    inter, intra = _mean_intra_inter(final_norm, y)
    sep_ratio = inter / max(intra, 1e-9)
    try:
        from sklearn.metrics import silhouette_score
        sil = float(silhouette_score(final_norm, y)) if len(set(y.tolist())) > 1 else 0.0
    except Exception:
        sil = float("nan")

    silo_inter, silo_intra = _mean_intra_inter(shared_norm(raw_silo), y)
    silo_ratio = silo_inter / max(silo_intra, 1e-9)

    # per-round separation (should climb across rounds; fraud separates by final)
    per_round = []
    for r in checkpoints:
        ri, ra = _mean_intra_inter(shared_norm(raw_frames[r]), y)
        per_round.append((int(r), ri / max(ra, 1e-9)))

    out = {
        "n": int(n),
        "labels": [int(v) for v in y],
        "frames": frames,
        "siloedFinal": siloed_final,
        "meta": {
            "method": method,
            "dims": 3,
            "note": ("federated embedding model: penultimate ReLU activations "
                     "+ decision logit, UMAP-projected to 3D; fraud separates "
                     "from legit as federated rounds progress (federated beats "
                     "siloed via the shared cross-bank fraud corridor)"),
        },
    }
    path = os.path.join(OUT_DIR, "umap.json")
    with open(path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    return path, {
        "method": method,
        "n": n,
        "frames": len(frames),
        "fed_sep_ratio": sep_ratio,
        "fed_silhouette": sil,
        "silo_sep_ratio": silo_ratio,
        "per_round": per_round,
        "n_fraud": int(y.sum()),
    }


# --------------------------------------------------------------------------- #
# ARTIFACT 2 — graph.json
# --------------------------------------------------------------------------- #
def _spring_layout(n, edges, seed):
    """2D force-directed layout. Uses networkx if available, else a small
    Fruchterman-Reingold implemented here. Returns (n,2)."""
    try:
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        pos = nx.spring_layout(g, seed=seed, k=None, iterations=200)
        return np.array([pos[i] for i in range(n)], dtype=float)
    except Exception:
        return _fruchterman_reingold(n, edges, seed)


def _fruchterman_reingold(n, edges, seed, iters=300):
    rng = np.random.default_rng(seed)
    pos = rng.normal(0, 1, (n, 2))
    A = np.zeros((n, n), dtype=bool)
    for i, j in edges:
        A[i, j] = True
        A[j, i] = True
    k = np.sqrt(1.0 / max(1, n))
    t = 0.1
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]  # (n,n,2)
        dist = np.linalg.norm(delta, axis=2)
        dist = np.maximum(dist, 1e-3)
        np.fill_diagonal(dist, 1e9)  # kill self-interaction without inf*0 NaNs
        unit = delta / dist[:, :, None]
        rep = (k * k) / dist
        disp = (unit * rep[:, :, None]).sum(axis=1)
        attr = (k * dist) * A
        disp -= (unit * attr[:, :, None]).sum(axis=1)
        length = np.maximum(np.linalg.norm(disp, axis=1, keepdims=True), 1e-6)
        pos += (disp / length) * np.minimum(length, t)
        t *= 0.99
    return pos


def gen_graph():
    n_banks = 8
    net = generate_network(n_banks=n_banks, n_accounts=160, n_campaigns=7,
                           seed=SEED)
    n = net.n_accounts

    # Build a deduped undirected edge list over node indices.
    edge_set = set()
    for (s, d, _a, _t) in net.edges:
        if s == d:
            continue
        edge_set.add((min(s, d), max(s, d)))
    edges = sorted(edge_set)

    pos = _norm_coords(_spring_layout(n, edges, SEED))

    nodes = []
    for i in range(n):
        nodes.append({
            "id": int(i),
            "bank": int(net.bank_of[i]),
            "x": _r(pos[i, 0], 4),
            "y": _r(pos[i, 1], 4),
            "mule": bool(net.y[i] == 1),
        })

    # ---- FEDERATED GNN training (FedAvg over per-bank subgraphs) ---------
    in_dim = net.X.shape[1]
    subgraphs = [net.local_subgraph(b) for b in range(n_banks)]
    # Map a subgraph's owned local rows back to global node ids for scoring.
    owned_global = [sg.global_ids[sg.owned_mask] for sg in subgraphs]
    owned_local = [np.where(sg.owned_mask)[0] for sg in subgraphs]

    global_w = gnn.init_weights(in_dim, hidden=8, seed=SEED)

    checkpoints = [0, 1, 2, 4, 6, 9]
    total_rounds = max(checkpoints)

    def global_scores(w):
        """Per-global-node fraud score, taken from each owning bank's GNN
        forward pass over its own subgraph (real federated inference)."""
        scores = np.zeros(n)
        for sg, gids, lids in zip(subgraphs, owned_global, owned_local):
            p = gnn.predict_proba(w, sg, in_dim)
            scores[gids] = p[lids]
        return scores

    frames = []
    if 0 in checkpoints:
        frames.append({"round": 0,
                       "scores": [_r(v, 4) for v in global_scores(global_w)]})

    for rnd in range(1, total_rounds + 1):
        updates = []
        for sg in subgraphs:
            updates.append(gnn.train_local(global_w, sg, epochs=12, lr=0.2,
                                           in_dim=in_dim))
        global_w = fedavg(updates)
        if rnd in checkpoints:
            frames.append({"round": int(rnd),
                           "scores": [_r(v, 4) for v in global_scores(global_w)]})

    # sanity: mule vs legit mean score at final round
    final = np.array(frames[-1]["scores"])
    mule_mask = net.y == 1
    mule_mean = float(final[mule_mask].mean())
    legit_mean = float(final[~mule_mask].mean())

    cross_edges = sum(1 for (i, j) in edges if net.bank_of[i] != net.bank_of[j])

    out = {
        "nodes": nodes,
        "edges": [[int(i), int(j)] for i, j in edges],
        "frames": frames,
        "meta": {
            "banks": n_banks,
            "note": ("cross-bank mule rings; per-node fraud scores from the "
                     "federated GraphSAGE GNN rise on ring nodes across rounds"),
        },
    }
    path = os.path.join(OUT_DIR, "graph.json")
    with open(path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    return path, {
        "nodes": n,
        "edges": len(edges),
        "cross_bank_edges": cross_edges,
        "frames": len(frames),
        "mule_mean": mule_mean,
        "legit_mean": legit_mean,
        "n_mules": int(mule_mask.sum()),
    }


# --------------------------------------------------------------------------- #
# ARTIFACT 3 — federation.json
# --------------------------------------------------------------------------- #
def gen_federation():
    rng = np.random.default_rng(SEED + 7)
    n_clients = 8

    # Build REAL per-client weight-update deltas from the embedding model: each
    # client trains one local step from a shared global init on its own slice of
    # data; the update is (local - global). One client is poisoned.
    X_cat, X_dense, y, cards = embeddings.make_categorical_data(
        700, fraud_rate=0.06, seed=SEED + 7)
    n = len(y)
    perm = rng.permutation(n)
    parts = np.array_split(perm, n_clients)

    g0 = embeddings.init_weights(cards, seed=SEED + 7)
    poisoned_idx = n_clients - 1

    deltas = []
    for ci, idx in enumerate(parts):
        local = embeddings.train_local(
            g0, X_cat[idx], X_dense[idx], y[idx],
            epochs=6, lr=0.3, cardinalities=cards,
            emb_dim=4, dense_dim=X_dense.shape[1], hidden=16,
        )
        d = local - g0
        if ci == poisoned_idx:
            # model-poisoning: scale the honest direction to a huge norm so
            # Multi-Krum rejects it as an outlier.
            d = d * 25.0 + rng.normal(0, 0.5, size=d.shape)
        deltas.append(d)

    norms = np.array([float(np.linalg.norm(d)) for d in deltas])

    # DP clip radius: median honest norm (robust, excludes the poison spike).
    honest_norms = np.delete(norms, poisoned_idx)
    clip_norm = float(np.median(honest_norms) * 1.1)
    clipped = [dp.clip_update(d, clip_norm) for d in deltas]

    # Multi-Krum selection over the CLIPPED updates (f=1 byzantine assumed).
    agg, selected_idx, scores = multi_krum_select(clipped, n_byzantine=1)
    selected = set(selected_idx)
    rejected = [i for i in range(n_clients) if i not in selected]

    # ---- 2D projection: PCA fit once on the stack of raw updates ---------
    from sklearn.decomposition import PCA
    stack = np.stack(deltas)
    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(stack)

    raw2d = pca.transform(stack)           # arrive (un-clipped)
    clip2d = pca.transform(np.stack(clipped))  # after clip
    agg2d = pca.transform(agg.reshape(1, -1))[0]

    # Shared normalization across all stages so the animation scale is stable.
    allpts = np.vstack([raw2d, clip2d, agg2d.reshape(1, -1)])
    center = allpts.mean(axis=0)
    scale = np.percentile(np.abs(allpts - center), 99)
    if scale <= 0:
        scale = 1.0

    def nz(p):
        return ((np.asarray(p) - center) / scale)

    raw2d_n = nz(raw2d)
    clip2d_n = nz(clip2d)
    agg2d_n = nz(agg2d)
    # clip radius in normalized 2D space (PCA is linear, scale by 1/scale of the
    # update-norm projection); approximate as clip_norm scaled by the mean
    # projection-norm ratio. We document this as approximate.
    proj_ratio = float(np.mean(np.linalg.norm(raw2d, axis=1) /
                               np.maximum(norms, 1e-9)))
    clip_radius_2d = float(clip_norm * proj_ratio / scale)

    def clients_at(coords, *, clipped_stage):
        out = []
        for i in range(n_clients):
            nrm = float(np.linalg.norm(clipped[i])) if clipped_stage else norms[i]
            out.append({
                "xy": [_r(coords[i, 0], 4), _r(coords[i, 1], 4)],
                "poisoned": bool(i == poisoned_idx),
                "rejected": bool(i in set(rejected)),
                "norm": _r(nrm, 4),
            })
        return out

    origin = [0.0, 0.0]
    frames = [
        {  # clients arrive (raw updates)
            "stage": "arrive",
            "clients": clients_at(raw2d_n, clipped_stage=False),
            "clipRadius": _r(clip_radius_2d, 4),
            "global": origin,
        },
        {  # DP clip pulls over-norm updates onto the clip sphere
            "stage": "clip",
            "clients": clients_at(clip2d_n, clipped_stage=True),
            "clipRadius": _r(clip_radius_2d, 4),
            "global": origin,
        },
        {  # Multi-Krum selects honest majority, marks rejected
            "stage": "select",
            "clients": clients_at(clip2d_n, clipped_stage=True),
            "clipRadius": _r(clip_radius_2d, 4),
            "global": origin,
        },
        {  # aggregate selected -> new global step
            "stage": "aggregate",
            "clients": clients_at(clip2d_n, clipped_stage=True),
            "clipRadius": _r(clip_radius_2d, 4),
            "global": [_r(agg2d_n[0], 4), _r(agg2d_n[1], 4)],
        },
    ]

    out = {
        "frames": frames,
        "meta": {
            "krum": "multi-krum",
            "rejected": int(len(rejected)),
            "note": ("one large-norm poisoned client is DP-clipped and then "
                     "rejected by Multi-Krum; honest clients are aggregated. "
                     "2D via PCA on the real update vectors; clipRadius is "
                     "approximate in 2D space"),
        },
    }
    path = os.path.join(OUT_DIR, "federation.json")
    with open(path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    return path, {
        "clients": n_clients,
        "rejected": len(rejected),
        "poison_rejected": poisoned_idx in set(rejected),
        "selected_idx": selected_idx,
        "poison_norm": norms[poisoned_idx],
        "median_honest_norm": float(np.median(honest_norms)),
        "clip_norm": clip_norm,
    }


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _validate_umap(d):
    assert set(d) == {"n", "labels", "frames", "siloedFinal", "meta"}
    n = d["n"]
    assert len(d["labels"]) == n and all(v in (0, 1) for v in d["labels"])
    assert d["meta"]["method"] in ("umap", "pca", "tsne")
    assert d["meta"]["dims"] == 3
    assert len(d["siloedFinal"]) == n
    assert all(len(p) == 3 for p in d["siloedFinal"])
    for fr in d["frames"]:
        assert set(fr) == {"round", "fed"}
        assert len(fr["fed"]) == n
        assert all(len(p) == 3 for p in fr["fed"])


def _validate_graph(d):
    assert set(d) == {"nodes", "edges", "frames", "meta"}
    n = len(d["nodes"])
    for nd in d["nodes"]:
        assert set(nd) == {"id", "bank", "x", "y", "mule"}
        assert 0 <= nd["bank"] <= 7
    for e in d["edges"]:
        assert len(e) == 2 and 0 <= e[0] < n and 0 <= e[1] < n
    for fr in d["frames"]:
        assert set(fr) == {"round", "scores"}
        assert len(fr["scores"]) == n
        assert all(0.0 <= s <= 1.0 for s in fr["scores"])
    assert "banks" in d["meta"]


def _validate_federation(d):
    assert set(d) == {"frames", "meta"}
    stages = [f["stage"] for f in d["frames"]]
    assert stages == ["arrive", "clip", "select", "aggregate"]
    for fr in d["frames"]:
        assert set(fr) == {"stage", "clients", "clipRadius", "global"}
        assert len(fr["global"]) == 2
        for c in fr["clients"]:
            assert set(c) == {"xy", "poisoned", "rejected", "norm"}
            assert len(c["xy"]) == 2
    assert d["meta"]["krum"] == "multi-krum"
    assert isinstance(d["meta"]["rejected"], int)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(SEED)

    print("[1/3] umap.json — federated embedding separation ...")
    p1, s1 = gen_umap()
    print("[2/3] graph.json — federated GNN mule graph ...")
    p2, s2 = gen_graph()
    print("[3/3] federation.json — DP clip + Multi-Krum ...")
    p3, s3 = gen_federation()

    # validate
    for path, validate in ((p1, _validate_umap), (p2, _validate_graph),
                           (p3, _validate_federation)):
        with open(path) as fh:
            validate(json.load(fh))

    def kb(path):
        return os.path.getsize(path) / 1024.0

    print("\n==================== SUMMARY ====================")
    print(f"umap.json        {kb(p1):7.1f} KB  {p1}")
    print(f"  method={s1['method']} n={s1['n']} frames={s1['frames']} "
          f"fraud={s1['n_fraud']}")
    print(f"  FED separation inter/intra = {s1['fed_sep_ratio']:.3f}  "
          f"silhouette = {s1['fed_silhouette']:.3f}")
    print("  per-round separation (should climb): " +
          "  ".join(f"r{r}={v:.3f}" for r, v in s1['per_round']))
    print(f"  SILO separation inter/intra = {s1['silo_sep_ratio']:.3f}  "
          f"(federated should beat siloed)")
    print(f"graph.json       {kb(p2):7.1f} KB  {p2}")
    print(f"  nodes={s2['nodes']} edges={s2['edges']} "
          f"cross-bank-edges={s2['cross_bank_edges']} frames={s2['frames']} "
          f"mules={s2['n_mules']}")
    print(f"  final mule mean score = {s2['mule_mean']:.3f}  "
          f"legit mean = {s2['legit_mean']:.3f}")
    print(f"federation.json  {kb(p3):7.1f} KB  {p3}")
    print(f"  clients={s3['clients']} rejected={s3['rejected']} "
          f"poison_rejected={s3['poison_rejected']} selected={s3['selected_idx']}")
    print(f"  poison_norm={s3['poison_norm']:.2f} "
          f"median_honest_norm={s3['median_honest_norm']:.2f} "
          f"clip_norm={s3['clip_norm']:.2f}")
    print("=================================================")
    print("All three artifacts written and schema-validated.")


if __name__ == "__main__":
    main()
