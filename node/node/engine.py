"""Single-bank node engine.

This is the split-out of ``core/veritas_core/engine.py`` (which simulated N
banks in one process) down to ONE bank. It holds:

  * ``global_w``  — the federated global model (synced from the control plane),
                    a packed ``live_ensemble`` vector (dim 617).
  * ``silo_w``    — the SILOED baseline: the SAME LiveEnsemble trained ONLY on
                    this bank's local data, sharing off. This is the honest
                    federated-vs-siloed counterfactual surfaced in /state.
  * ``edge_w``    — the bank EDGE model (Tier-0 device updates aggregate into it).
                    KEPT as the flat logistic (dim 11) — the TypeScript edge SDK
                    sends dim-11 updates, so the device path is unchanged.

Local MULTI-VIEW data (``LiveData``: tabular X + categorical X_cat) comes from
the connector runtime (config-driven, lifted into a neutral categorical view)
and falls back to the synthetic ``live_ensemble.make_live_data`` generator when
no real source is wired, so the node is runnable and testable standalone. All
FL math is reused from core.
"""
from __future__ import annotations

import threading
import uuid

import numpy as np

from veritas_core.data import FEATURE_DIM
from veritas_core.gnn_benchmark import benchmark_current, compute_gnn_benchmark
from veritas_core import live_ensemble
from veritas_core.live_ensemble import LiveData
# Edge (Tier-0) model stays the flat logistic (dim FEATURE_DIM+1==11): the
# TypeScript edge SDK sends dim-11 updates, so the device path is unchanged.
from veritas_core.model import init_weights as edge_init_weights
from veritas_core.secure_agg import (
    establish_pairwise_seeds,
    recover_dropout,
    secure_sum,
)

from .config import EPOCHS, LR, NodeConfig

# Business constants mirror core/veritas_core/engine.py so the single-bank
# counterfactual reads consistently with the multi-bank demo.
AVG_LOSS = 255
AT_RISK = 1500
THRESH = 0.9
HOURS = 1.0

NAMES = ["Barclays", "NatWest", "Lloyds", "HSBC", "Santander", "Monzo", "Starling", "Nationwide"]
CUST = [2_100_000, 1_900_000, 1_750_000, 1_600_000, 1_400_000, 900_000, 700_000, 1_500_000]


class NodeEngine:
    def __init__(self, cfg: NodeConfig, *, connector_data: tuple[np.ndarray, np.ndarray] | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._lock = threading.RLock()

        i = cfg.node_index
        self.name = NAMES[i % len(NAMES)]
        self.customers = CUST[i % len(CUST)]

        # Local MULTI-VIEW training data (LiveData: tabular X + categorical X_cat)
        # for the stacked LiveEnsemble. Prefer the connector (config-driven); the
        # connector yields a flat (X, y) tabular view, which we lift into a
        # LiveData with a neutral categorical view (its fraud typology is tabular).
        # When no connector is wired we synthesise the heterogeneous multi-view
        # sample so the ensemble's categorical/XOR typologies are present too.
        # A designated BLIND node must train on campaign-FREE data so its siloed
        # baseline can never learn the new typology (only the federated model,
        # via seeing peers, can) — mirrors the live deploy, where the blind node
        # is pointed at a nonexistent feature map to force the synthetic
        # campaign-free fallback. So we ignore connector data for a blind node,
        # whose sample set may itself carry campaign-signature rows.
        is_blind = cfg.blind_node is not None and cfg.blind_node == cfg.node_index
        if connector_data is not None and len(connector_data[1]) > 0 and not is_blind:
            X, y = connector_data
            self.train_data, self.train_y = self._wrap_tabular(X), y
        else:
            self.train_data, self.train_y = self._make_train(cfg.seed)
        # Held-out eval set for the recall counterfactual.
        self.eval_data, self.eval_y = self._make_eval(cfg.seed)

        # Federated + siloed models are the packed LiveEnsemble (dim 617). The
        # edge model stays the flat logistic (dim 11) — the device SDK contract.
        self.global_w = live_ensemble.init_weights(seed=cfg.seed)   # federated
        self.silo_w = live_ensemble.init_weights(seed=cfg.seed)     # siloed
        self.edge_w = edge_init_weights(FEATURE_DIM)                 # edge (device→bank)

        self.global_version = 0
        self.round = 0
        self.campaign_active = False
        self.cum = {"fed": 0.0, "silo": 0.0}

        # ---- edge secure-aggregation (Bonawitz pairwise masking) state -----
        # The bank node is the cohort dealer/relay (production: relays X25519
        # public keys; here it deals the per-pair seeds). For an OPEN cohort it
        # buffers ONLY the masked device messages it receives — it never sees,
        # stores, or can derive any individual device's cleartext update. On
        # close it secure-sums the masked messages (recovering dropouts) and
        # folds the single aggregate into ``edge_w``.
        self._cohort_id: str | None = None
        self._cohort_client_ids: list[str] = []          # planned cohort membership
        self._cohort_seed_table: dict = {}               # per-pair seeds (dealer)
        self._cohort_masked: dict[str, np.ndarray] = {}  # clientId -> masked vector
        self._edge_cohorts_aggregated = 0                # closed-cohort counter
        self._edge_updates_seen = 0                      # device messages buffered

        # Real federated-GNN mule-graph benchmark (the headline siloed-vs-federated
        # contrast). It TRAINS a numpy GNN, which takes a few seconds, so compute
        # it ONCE in a background thread at startup; until ready state() omits
        # gnnBenchmark (the web shows "GNN pending"). The live `current` block is
        # built per state() call from the REAL round trajectory, keyed on this
        # node's round / campaign flag.
        self._gnn_benchmark: dict | None = None
        threading.Thread(target=self._compute_gnn_benchmark, daemon=True).start()

    # ---- multi-view data builders (LiveData for the ensemble) ------------

    # Sizes kept modest so the heavier ensemble train_local stays snappy per
    # round in the live demo (the stacked model is far costlier than logistic).
    _TRAIN_N = 1200
    _EVAL_N = 600

    @staticmethod
    def _wrap_tabular(X: np.ndarray) -> LiveData:
        """Lift a flat (n, FEATURE_DIM) tabular matrix into a LiveData with a
        neutral categorical view (category 0 everywhere — NOT the mule corridor).
        """
        X = np.asarray(X, dtype=np.float64)
        X_cat = np.zeros((X.shape[0], 3), dtype=np.int64)
        return LiveData(X=X, X_cat=X_cat)

    def _make_train(self, seed: int, *, inject_campaign: bool = False):
        return live_ensemble.make_live_data(
            self._TRAIN_N, fraud_rate=0.03, seed=seed,
            inject_campaign=inject_campaign)

    def _make_eval(self, seed: int, *, inject_campaign: bool = False):
        return live_ensemble.make_live_data(
            self._EVAL_N, fraud_rate=0.05, seed=seed + 100,
            inject_campaign=inject_campaign)

    @staticmethod
    def _append_campaign(data: LiveData, y: np.ndarray, n_camp: int, seed: int):
        """Append a "safe-account mule" campaign block to an existing multi-view
        sample (mirrors live_ensemble.make_live_data's inject_campaign block).

        Returns (LiveData, y) with ``n_camp`` extra fraud rows that look benign on
        the generic fraud axes but carry the campaign signature in feature [-1].
        Appending (rather than regenerating) guarantees STRICTLY more positives
        for any base — connector-loaded or synthetic — so a "seeing" bank's
        siloed model gains local campaign examples and a targeted eval set grows.
        """
        crng = np.random.default_rng(seed)
        # Campaign rows must look BENIGN on every axis a campaign-blind siloed
        # model knows (small noise around 0, the generic-fraud margins OFF), and
        # carry the campaign typology ONLY in the signature feature [-1]. A model
        # that never trained on campaign rows leaves feature[-1]'s weight ~0 and
        # cannot flag them; a FEDERATED model that learned the signature from
        # seeing peers can. This is the load-bearing siloed-vs-federated lift.
        Xc = crng.normal(0.0, 0.3, (n_camp, FEATURE_DIM))
        Xc[:, -1] = 1.5                              # campaign signature ONLY
        card = live_ensemble.CARD
        cat_c = np.stack([
            crng.integers(0, card[0], n_camp),
            crng.integers(0, card[1], n_camp),
            crng.integers(0, card[2], n_camp),
        ], axis=1).astype(np.int64)
        X = np.vstack([data.X, Xc.astype(np.float64)])
        X_cat = np.vstack([data.X_cat, cat_c])
        y = np.concatenate([np.asarray(y), np.ones(n_camp, dtype=np.int64)])
        return LiveData(X=X, X_cat=X_cat), y

    def _compute_gnn_benchmark(self) -> None:
        b = compute_gnn_benchmark(seed=0)
        with self._lock:
            self._gnn_benchmark = b

    # ---- campaign (demo toggle, same semantics as core) ------------------

    def inject_campaign(self, *, seeing: bool = True) -> None:
        """Start the cross-institution scam campaign for this bank.

        ``seeing=True`` (default): the campaign appears in BOTH this bank's local
        TRAINING data and its EVAL set — a bank that directly observes the new
        typology. Its siloed model can therefore learn it.

        ``seeing=False``: the campaign appears in the EVAL set ONLY — this bank's
        customers are TARGETED but it has no local campaign examples, so its
        SILOED model stays blind. Only the FEDERATED model (which learns the
        signature from other "seeing" banks via aggregation) flags it. This is
        the honest federated-vs-siloed lift: the same contrast core's multi-bank
        engine surfaces (eval targeted everywhere; training campaign only at a
        subset of banks).
        """
        with self._lock:
            self.campaign_active = True
            # Append the "safe-account mule" campaign block (make_live_data's
            # campaign typology) to the existing multi-view data. seeing ->
            # campaign in BOTH local training and eval (the bank can learn it);
            # blind -> campaign in EVAL ONLY (targeted but the siloed model never
            # trains on it, so only the FEDERATED model carries the signature in
            # from seeing peers). Appending guarantees strictly more positives.
            if seeing:
                self.train_data, self.train_y = self._append_campaign(
                    self.train_data, self.train_y, 150, seed=1 + self.cfg.node_index)
            self.eval_data, self.eval_y = self._append_campaign(
                self.eval_data, self.eval_y, 80, seed=200 + self.cfg.node_index)

    def reset_genesis(self) -> None:
        """Reset local FL state to genesis so the demo RACE can be re-run.

        Re-initialises the federated/siloed/edge weights, clears the campaign
        (fresh training + eval data, sharing off), and zeroes the running
        counters. Called when the control plane bumps its ``epoch`` (reset
        beat). The connector-loaded training data is regenerated to its pristine
        (campaign-free) form so a blind node is blind again on the next run.
        """
        with self._lock:
            # Pristine training/eval data (mirrors __init__ synthetic fallback;
            # connector reloads would re-fetch upstream, which the live demo
            # resets via a fresh process — here we regenerate the synthetic
            # baseline so re-runs are deterministic and campaign-free).
            self.train_data, self.train_y = self._make_train(self.cfg.seed)
            self.eval_data, self.eval_y = self._make_eval(self.cfg.seed)
            self.global_w = live_ensemble.init_weights(seed=self.cfg.seed)
            self.silo_w = live_ensemble.init_weights(seed=self.cfg.seed)
            self.edge_w = edge_init_weights(FEATURE_DIM)
            self.global_version = 0
            self.round = 0
            self.campaign_active = False
            self.cum = {"fed": 0.0, "silo": 0.0}

    # ---- model sync from the federation client ---------------------------

    def set_global(self, weights: np.ndarray, version: int) -> None:
        with self._lock:
            self.global_w = np.asarray(weights, dtype=np.float64)
            self.global_version = int(version)
            self.round = max(self.round, version)

    def train_silo_step(self) -> None:
        """Advance the siloed ENSEMBLE baseline one local step (sharing off)."""
        with self._lock:
            self.silo_w = live_ensemble.train_local(
                self.silo_w, self.train_data, self.train_y, epochs=EPOCHS, lr=LR)

    # ---- detection / counterfactual --------------------------------------

    def _det(self) -> tuple[float, float]:
        fed = live_ensemble.recall(self.global_w, self.eval_data, self.eval_y)
        silo = live_ensemble.recall(self.silo_w, self.eval_data, self.eval_y)
        return float(fed), float(silo)

    def silo_recall(self) -> float:
        """Measured recall of the SILOED baseline on the held-out eval set.

        This is the honest siloed counterfactual the node reports as
        ``localMetrics.siloRecall`` (advance ``train_silo_step`` first so it is
        meaningful).
        """
        with self._lock:
            return float(
                live_ensemble.recall(self.silo_w, self.eval_data, self.eval_y))

    def predict(self, features: np.ndarray, *, weights: np.ndarray | None = None,
                edge: bool = False) -> tuple[str, float]:
        """Score one transaction.

        Federated/global predictions run the packed LiveEnsemble via
        ``predict_one`` (a bare FEATURE_DIM row -> neutral categorical view). The
        EDGE path keeps the flat logistic (dim 11), so ``edge=True`` scores with
        ``model.predict_proba`` on ``edge_w``.
        """
        if edge or (weights is not None and len(np.asarray(weights)) == self.edge_w.shape[0]):
            from veritas_core.model import predict_proba as _edge_proba
            w = self.edge_w if weights is None else weights
            p = float(_edge_proba(np.asarray(w), np.asarray(features).reshape(1, -1))[0])
        else:
            w = self.global_w if weights is None else weights
            p = float(live_ensemble.predict_one(w, np.asarray(features)))
        return ("fraud" if p >= 0.5 else "legitimate", p)

    # ---- edge secure-aggregation (Tier 0 → bank, Bonawitz masked sum) -----

    def open_cohort(self, client_ids: list[str], *, cohort_id: str | None = None,
                    master_rng: np.random.Generator | None = None) -> dict:
        """Open a secure-aggregation cohort and deal pairwise seeds.

        The node plays cohort dealer/relay: it fixes the cohort membership and
        establishes the pairwise seed table. In PRODUCTION the node only relays
        device X25519 public keys and the per-pair seeds ``s_ij`` are derived
        independently by each device via authenticated Diffie-Hellman, so the
        node NEVER learns a seed and is structurally unable to unmask an
        individual. THIS REFERENCE deals the seeds locally (trusted-dealer
        stand-in) so the masking algebra runs end-to-end without a crypto dep.

        Each device receives its ``clientId``, peer list, and the seeds for the
        pairs that include it. Returns the dealer assignment table.
        """
        with self._lock:
            ids = list(client_ids)
            if len(set(ids)) != len(ids):
                raise ValueError("cohort client_ids must be unique")
            self._cohort_id = cohort_id or uuid.uuid4().hex
            self._cohort_client_ids = ids
            rng = master_rng if master_rng is not None else self.rng
            self._cohort_seed_table = establish_pairwise_seeds(ids, master_rng=rng)
            self._cohort_masked = {}
            return {
                "cohortId": self._cohort_id,
                "clientIds": ids,
                # per-pair seeds, exposed so each device can mask against peers
                "seedTable": {f"{a}|{b}": s for (a, b), s in self._cohort_seed_table.items()},
            }

    def edge_aggregate(self, update: np.ndarray, num_examples: int,
                       *, cohort_id: str | None = None, client_id: str | None = None) -> None:
        """Buffer ONE device's MASKED update for the open cohort.

        The bank node never sees an individual device's cleartext update: the
        device has already clipped + DP-noised its delta CLIENT-SIDE and then
        applied Bonawitz pairwise masks. We only store the masked vector keyed
        by ``client_id``; the masks make it look uniformly random. DP is NOT
        applied here (it happened on-device before masking) — the node only
        ever sums. The cohort is closed (secure-summed and folded) by
        ``close_cohort``.

        ``num_examples`` is accepted for API compatibility but no longer used to
        re-weight a single message (weighting is uniform within a secure sum).

        If no cohort is open we open a singleton-cohort for this client so the
        masked message is still never un-masked individually; with one client
        the secure sum is the (DP) update itself, which is the legacy behaviour.
        """
        with self._lock:
            u = np.asarray(update, dtype=np.float64)
            if u.shape != self.edge_w.shape:
                raise ValueError(f"edge update dim {u.shape} != model dim {self.edge_w.shape}")
            cid = client_id or f"dev-{len(self._cohort_masked)}"
            if self._cohort_id is None:
                # Auto-open a degenerate cohort (no peers => mask is the update).
                self.open_cohort([cid], cohort_id=cohort_id)
            if cohort_id is not None and cohort_id != self._cohort_id:
                raise ValueError(
                    f"update for cohort {cohort_id!r} but open cohort is {self._cohort_id!r}")
            # Store ONLY the masked vector. Never store/derive the cleartext.
            self._cohort_masked[cid] = u
            self._edge_updates_seen += 1

    def close_cohort(self) -> np.ndarray:
        """Secure-sum the buffered masked messages and fold the aggregate in.

        Computes ``secure_sum`` over the masked messages actually received. If
        some planned cohort members dropped out (never submitted), their
        pairwise masks no longer cancel; ``recover_dropout`` subtracts those
        leftover masks (production: their seeds are reconstructed via Shamir from
        surviving shares; here from the dealer's seed table) so the result is the
        true SUM over the SURVIVORS only. The single aggregate — never any
        individual update — is then folded into ``edge_w``.

        Returns the recovered aggregate (for tests/inspection).
        """
        with self._lock:
            if not self._cohort_masked:
                self._reset_cohort()
                return np.zeros_like(self.edge_w)
            survivors = list(self._cohort_masked.keys())
            masked_vecs = [self._cohort_masked[c] for c in survivors]
            agg = secure_sum(masked_vecs)
            planned = self._cohort_client_ids or survivors
            dropped = [c for c in planned if c not in self._cohort_masked]
            if dropped:
                agg = recover_dropout(
                    agg, dropped, survivors, self._cohort_seed_table, dim=self.edge_w.shape[0])
            # Fold the single aggregate into the edge model. Normalise by the
            # number of surviving contributors so the step is scale-stable.
            n = max(1, len(survivors))
            self.edge_w = self.edge_w + agg / n
            self._edge_cohorts_aggregated += 1
            self._reset_cohort()
            return agg

    def _reset_cohort(self) -> None:
        self._cohort_id = None
        self._cohort_client_ids = []
        self._cohort_seed_table = {}
        self._cohort_masked = {}

    @property
    def cohort_open(self) -> bool:
        return self._cohort_id is not None

    def edge_model(self) -> dict:
        with self._lock:
            return {"version": self.global_version, "dim": self.cfg.dim, "weights": self.edge_w.tolist()}

    # ---- state (single-bank, federated-vs-siloed) ------------------------

    def state(self) -> dict:
        with self._lock:
            fed, silo = self._det()
            # Cumulative victim/loss counters, single-bank flavour of core.
            self.cum_snapshot = None
            fv = int(self.cum["fed"])
            sv = int(self.cum["silo"])
            ttd = lambda a: HOURS * max(1, self.round) if a >= THRESH else 101.0
            bank = {
                "id": self.cfg.node_id,
                "name": self.name,
                "customers": self.customers,
                "detection": {"federated": round(fed, 3), "siloed": round(silo, 3)},
                "poisoned": False,
            }
            state = {
                "round": self.round,
                "running": True,
                "banks": [bank],
                "campaignActive": self.campaign_active,
                "attackActive": False,
                "customerRecordsTransmitted": 0,
                "node": {
                    "memberId": self.cfg.node_id,
                    "tenantId": self.cfg.tenant_id,
                    "modelVersion": self.global_version,
                    "edgeUpdatesAggregated": self._edge_updates_seen,
                    "edgeCohortsAggregated": self._edge_cohorts_aggregated,
                    "cohortOpen": self._cohort_id is not None,
                },
                "counters": {
                    "federated": {
                        "fraudPreventedGbp": max(0, sv - fv) * AVG_LOSS,
                        "timeToDetectHours": ttd(fed),
                        "victims": fv,
                        "lostGbp": fv * AVG_LOSS,
                    },
                    "siloed": {
                        "fraudPreventedGbp": 0,
                        "timeToDetectHours": ttd(silo),
                        "victims": sv,
                        "lostGbp": sv * AVG_LOSS,
                    },
                },
            }
            if self._gnn_benchmark is not None:
                b = dict(self._gnn_benchmark)
                b["current"] = benchmark_current(
                    self._gnn_benchmark, self.round, self.campaign_active)
                state["gnnBenchmark"] = b
            return state

    def accrue_counters(self) -> None:
        """Advance the running victim/loss totals one round (federated vs siloed)."""
        with self._lock:
            fed, silo = self._det()
            self.cum["fed"] += AT_RISK * (1 - fed)
            self.cum["silo"] += AT_RISK * (1 - silo)
