"""Synthetic CROSS-BANK transaction graph generator for federated GNN fraud
detection.

APP / mule fraud is a GRAPH problem: a single collector account fans IN from
many victims and fans OUT (layering) to a spray of mule accounts, and those
mules deliberately sit at *other* banks so no single institution sees the whole
"safe account" topology. This module builds that global graph, partitions
accounts across N banks, and exposes per-bank *local subgraphs* in which
cross-bank neighbours appear only as feature-less BOUNDARY nodes (a bank sees
the transaction edge but not the counterparty bank's private node features).

NUMPY ONLY. Nodes carry a fixed-width feature vector; edges are directed with
an amount and timestamp. Mule accounts (collectors + downstream mules) are
labelled 1, legitimate background accounts 0.
"""
import numpy as np

# Node feature layout (NODE_FEATURE_DIM dims). Chosen to be the structural /
# behavioural signals a GNN can sharpen by aggregating over neighbours:
#   0 account_age        (lower for freshly-minted mules)
#   1 log_in_velocity    (incoming tx rate)
#   2 log_out_velocity   (outgoing tx rate)
#   3 balance_churn      (turnover / balance -> ~0 holding for pass-through mules)
#   4 in_degree          (normalised)
#   5 out_degree         (normalised)
#   6 mean_in_amount
#   7 mean_out_amount
#   8 amount_symmetry    (|in-out| flow imbalance; ~0 for clean pass-through)
#   9 cross_bank_ratio   (fraction of counterparties at OTHER banks)
NODE_FEATURE_DIM = 10

# Feature index that is DELIBERATELY private/structural so that a single siloed
# bank cannot reconstruct it (cross_bank_ratio depends on global topology).
CROSS_BANK_RATIO = 9


class Subgraph:
    """A single bank's view of the global graph.

    Attributes
    ----------
    X : (n_nodes, NODE_FEATURE_DIM) float64
        Node features. Boundary (foreign-bank) nodes have their private features
        ZEROED — the bank sees that the edge exists but not the counterparty's
        behavioural profile.
    y : (n_nodes,) int64
        Mule label (1) / legit (0). Boundary node labels are present for the
        *background topology* but are masked out of training via `owned_mask`.
    A : (n_nodes, n_nodes) float64
        Dense adjacency over local + boundary nodes (symmetrised for message
        passing; edges to boundary nodes are retained so structure propagates).
    owned_mask : (n_nodes,) bool
        True for nodes this bank actually owns (and may train/evaluate on).
        False for boundary nodes.
    global_ids : (n_nodes,) int64
        Map back to global account ids (useful for tests / debugging).
    """

    def __init__(self, X, y, A, owned_mask, global_ids):
        self.X = X
        self.y = y
        self.A = A
        self.owned_mask = owned_mask
        self.global_ids = global_ids

    @property
    def n_nodes(self):
        return self.X.shape[0]

    def normalized_adj(self):
        """Symmetric-ish mean aggregation operator with self-loops:
        Â = D^{-1}(A + I). Row-normalised so message passing = mean of
        (self + neighbour) features. Returns dense (n, n)."""
        A = self.A + np.eye(self.n_nodes)
        deg = A.sum(axis=1, keepdims=True)
        deg[deg == 0] = 1.0
        return A / deg


class Network:
    """Global cross-bank transaction graph + bank partition."""

    def __init__(self, X, y, edges, bank_of, n_banks):
        self.X = X                  # (N, NODE_FEATURE_DIM)
        self.y = y                  # (N,)
        self.edges = edges          # list of (src, dst, amount, timestamp)
        self.bank_of = bank_of      # (N,) int: owning bank per account
        self.n_banks = n_banks

    @property
    def n_accounts(self):
        return self.X.shape[0]

    def cross_bank_edges(self):
        """Edges whose endpoints sit at different banks (the federation signal)."""
        return [(s, d, a, t) for (s, d, a, t) in self.edges
                if self.bank_of[s] != self.bank_of[d]]

    def local_subgraph(self, bank):
        """Return bank `bank`'s local Subgraph.

        Owned nodes = accounts at this bank. We additionally pull in BOUNDARY
        nodes: foreign accounts that share an edge with an owned account, so the
        bank can message-pass across the cut — but boundary node features are
        zeroed (private to the other bank).
        """
        owned = np.where(self.bank_of == bank)[0]
        owned_set = set(owned.tolist())

        # Discover boundary nodes via incident edges.
        boundary = set()
        incident = []
        for (s, d, a, t) in self.edges:
            s_owned = s in owned_set
            d_owned = d in owned_set
            if s_owned or d_owned:
                incident.append((s, d, a, t))
                if not s_owned:
                    boundary.add(s)
                if not d_owned:
                    boundary.add(d)

        local_ids = list(owned) + sorted(boundary)
        gid2local = {g: i for i, g in enumerate(local_ids)}
        n = len(local_ids)

        X = self.X[local_ids].copy()
        y = self.y[local_ids].copy()
        owned_mask = np.zeros(n, dtype=bool)
        owned_mask[:len(owned)] = True  # owned nodes are first by construction

        # Hide boundary (foreign-bank) private features: a bank sees the edge,
        # not the counterparty's behavioural profile.
        X[~owned_mask] = 0.0

        # Build adjacency (directed edges, but we symmetrise for aggregation).
        A = np.zeros((n, n), dtype=np.float64)
        for (s, d, a, t) in incident:
            li, lj = gid2local[s], gid2local[d]
            A[li, lj] = 1.0
            A[lj, li] = 1.0

        return Subgraph(X, y, A, owned_mask, np.array(local_ids, dtype=np.int64))


def _standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def generate_network(n_banks=4, n_accounts=600, n_campaigns=6, seed=0):
    """Build a cross-bank mule transaction graph.

    Background: legitimate accounts wired into a sparse random graph (mostly
    same-bank, occasional cross-bank legit payments).

    Campaigns: each injected mule campaign is a connected subgraph —
        victims  --(high fan-IN)-->  collector  --(rapid fan-OUT)-->  mules
    The collector and its downstream mules are deliberately placed at DIFFERENT
    banks (cross-bank layering), so the full ring only exists in the union of
    bank views. Collector + mules are labelled 1.
    """
    rng = np.random.default_rng(seed)
    N = n_accounts

    # ---- assign every account to a bank (round-robin-ish, randomised) ----
    bank_of = rng.integers(0, n_banks, size=N)

    # ---- raw behavioural feature scaffolding for background accounts ----
    age = rng.uniform(0.3, 1.0, N)            # mostly mature accounts
    in_vel = rng.gamma(2.0, 1.0, N)
    out_vel = rng.gamma(2.0, 1.0, N)
    churn = rng.uniform(0.2, 0.9, N)
    label = np.zeros(N, dtype=np.int64)

    edges = []  # (src, dst, amount, timestamp)

    # ---- background legitimate edges (sparse) ----
    n_bg_edges = N * 2
    for _ in range(n_bg_edges):
        s = int(rng.integers(0, N))
        d = int(rng.integers(0, N))
        if s == d:
            continue
        amt = float(rng.gamma(2.0, 50.0))
        ts = float(rng.uniform(0, 1000))
        edges.append((s, d, amt, ts))

    # ---- inject mule campaigns ----
    # We reserve pools of accounts to act as collectors/mules/victims.
    mule_accounts = set()
    for c in range(n_campaigns):
        # pick a collector at some bank
        collector = int(rng.integers(0, N))
        coll_bank = bank_of[collector]

        # victims (legit accounts) fanning IN to the collector
        n_victims = int(rng.integers(6, 12))
        victims = rng.integers(0, N, size=n_victims).tolist()

        # downstream mules placed at OTHER banks where possible (cross-bank)
        n_mules = int(rng.integers(5, 10))
        mules = []
        for _ in range(n_mules):
            # bias selection toward accounts NOT at the collector's bank
            cand = int(rng.integers(0, N))
            tries = 0
            while bank_of[cand] == coll_bank and tries < 8:
                cand = int(rng.integers(0, N))
                tries += 1
            mules.append(cand)

        base_ts = float(rng.uniform(0, 900))

        # high fan-IN: victims -> collector, large-ish amounts, clustered in time
        for v in victims:
            if v == collector:
                continue
            amt = float(rng.gamma(5.0, 200.0))
            edges.append((v, collector, amt, base_ts + float(rng.uniform(0, 2))))

        # rapid fan-OUT / layering: collector -> mules, near-equal split, moments later
        for m in mules:
            if m == collector:
                continue
            amt = float(rng.gamma(5.0, 180.0))
            edges.append((collector, m, amt, base_ts + float(rng.uniform(2, 6))))
            # second hop layering: mule -> another mule (deeper structure)
            if len(mules) > 1:
                m2 = int(rng.choice(mules))
                if m2 != m:
                    edges.append((m, m2, amt * 0.9,
                                  base_ts + float(rng.uniform(6, 10))))

        # label collector + mules as fraud and stamp mule-like behaviour
        ring = [collector] + mules
        for a in ring:
            mule_accounts.add(a)
            label[a] = 1
        # collectors: very high fan-in
        in_vel[collector] += 8.0
        out_vel[collector] += 8.0
        age[collector] = min(age[collector], 0.15)   # freshly minted
        churn[collector] = 0.02                        # pass-through, no holding
        # mules: high fan-out, low holding
        for m in mules:
            out_vel[m] += 5.0
            in_vel[m] += 3.0
            age[m] = min(age[m], 0.2)
            churn[m] = 0.03

    # ---- derive degree / amount / topology features from the edge list ----
    in_deg = np.zeros(N)
    out_deg = np.zeros(N)
    in_amt = np.zeros(N)
    out_amt = np.zeros(N)
    cross_cnt = np.zeros(N)
    deg_cnt = np.zeros(N)
    for (s, d, amt, ts) in edges:
        out_deg[s] += 1
        in_deg[d] += 1
        out_amt[s] += amt
        in_amt[d] += amt
        deg_cnt[s] += 1
        deg_cnt[d] += 1
        if bank_of[s] != bank_of[d]:
            cross_cnt[s] += 1
            cross_cnt[d] += 1

    mean_in = in_amt / np.maximum(in_deg, 1)
    mean_out = out_amt / np.maximum(out_deg, 1)
    symmetry = np.abs(in_amt - out_amt) / np.maximum(in_amt + out_amt, 1.0)
    cross_ratio = cross_cnt / np.maximum(deg_cnt, 1)

    feats = np.column_stack([
        age,
        np.log1p(in_vel),
        np.log1p(out_vel),
        churn,
        in_deg,
        out_deg,
        mean_in,
        mean_out,
        symmetry,
        cross_ratio,
    ]).astype(np.float64)

    X = _standardize(feats)

    return Network(X=X, y=label, edges=edges, bank_of=bank_of, n_banks=n_banks)
