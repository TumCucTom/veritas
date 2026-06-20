"""Single-bank node engine.

This is the split-out of ``core/veritas_core/engine.py`` (which simulated N
banks in one process) down to ONE bank. It holds:

  * ``global_w``  — the federated global model (synced from the control plane).
  * ``silo_w``    — the SILOED baseline: trained ONLY on this bank's local data,
                    sharing off. This is the honest federated-vs-siloed
                    counterfactual surfaced in /state and the console.
  * ``edge_w``    — the bank edge model that Tier-0 device updates aggregate into.

Local data comes from the connector runtime (config-driven) and falls back to
the synthetic ``make_bank_data`` generator when no real source is wired, so the
node is runnable and testable standalone. All FL math is reused from core.
"""
from __future__ import annotations

import threading
import uuid

import numpy as np

from veritas_core.data import FEATURE_DIM, inject_campaign, make_bank_data
from veritas_core.gnn_benchmark import benchmark_current, compute_gnn_benchmark
from veritas_core.model import init_weights, predict_proba, recall, train_local
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

        # Local training data: prefer the connector (config-driven); else synth.
        if connector_data is not None and len(connector_data[1]) > 0:
            self.train_X, self.train_y = connector_data
        else:
            self.train_X, self.train_y = make_bank_data(3000, 0.03, seed=cfg.seed)
        # Held-out eval set for the recall counterfactual.
        self.eval_X, self.eval_y = make_bank_data(1000, 0.05, seed=cfg.seed + 100)

        self.global_w = init_weights(FEATURE_DIM)   # federated (synced from plane)
        self.silo_w = init_weights(FEATURE_DIM)     # siloed baseline (local only)
        self.edge_w = init_weights(FEATURE_DIM)     # bank edge model (device→bank)

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
            if seeing:
                self.train_X, self.train_y = inject_campaign(
                    self.train_X, self.train_y, 150, seed=1 + self.cfg.node_index)
            self.eval_X, self.eval_y = inject_campaign(
                self.eval_X, self.eval_y, 80, seed=200 + self.cfg.node_index)

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
            self.train_X, self.train_y = make_bank_data(3000, 0.03, seed=self.cfg.seed)
            self.eval_X, self.eval_y = make_bank_data(1000, 0.05, seed=self.cfg.seed + 100)
            self.global_w = init_weights(FEATURE_DIM)
            self.silo_w = init_weights(FEATURE_DIM)
            self.edge_w = init_weights(FEATURE_DIM)
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
        """Advance the siloed baseline one local step (sharing off)."""
        with self._lock:
            self.silo_w = train_local(self.silo_w, self.train_X, self.train_y, epochs=EPOCHS, lr=LR)

    # ---- detection / counterfactual --------------------------------------

    def _det(self) -> tuple[float, float]:
        fed = recall(self.global_w, self.eval_X, self.eval_y)
        silo = recall(self.silo_w, self.eval_X, self.eval_y)
        return float(fed), float(silo)

    def silo_recall(self) -> float:
        """Measured recall of the SILOED baseline on the held-out eval set.

        This is the honest siloed counterfactual the node reports as
        ``localMetrics.siloRecall`` (advance ``train_silo_step`` first so it is
        meaningful).
        """
        with self._lock:
            return float(recall(self.silo_w, self.eval_X, self.eval_y))

    def predict(self, features: np.ndarray, *, weights: np.ndarray | None = None) -> tuple[str, float]:
        w = self.global_w if weights is None else weights
        p = float(predict_proba(w, features.reshape(1, -1))[0])
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
