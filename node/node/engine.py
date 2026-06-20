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

import numpy as np

from veritas_core.data import FEATURE_DIM, inject_campaign, make_bank_data
from veritas_core.dp import privatize
from veritas_core.model import init_weights, predict_proba, recall, train_local

from .config import EPOCHS, LR, MAX_NORM, NodeConfig, SIGMA

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
        # Edge secure-aggregation accumulator (DP-summed; per-device never stored)
        self._edge_updates_seen = 0

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

    def predict(self, features: np.ndarray, *, weights: np.ndarray | None = None) -> tuple[str, float]:
        w = self.global_w if weights is None else weights
        p = float(predict_proba(w, features.reshape(1, -1))[0])
        return ("fraud" if p >= 0.5 else "legitimate", p)

    # ---- edge secure-aggregation (Tier 0 → bank, in-tenancy) -------------

    def edge_aggregate(self, update: np.ndarray, num_examples: int) -> None:
        """Secure-aggregate ONE device update into the bank edge model with DP.

        We never store the per-device update: it is DP-privatized and folded
        straight into ``edge_w``. ``num_examples`` weights the step. This is the
        DP-only secure-aggregation posture (v1) from the spec; a Bonawitz-style
        masked sum is the production upgrade.
        """
        with self._lock:
            u = np.asarray(update, dtype=np.float64)
            if u.shape != self.edge_w.shape:
                raise ValueError(f"edge update dim {u.shape} != model dim {self.edge_w.shape}")
            priv = privatize(u, MAX_NORM, SIGMA, self.rng)
            # weight by example count, normalised against a nominal batch
            scale = min(1.0, max(1, num_examples) / 64.0)
            self.edge_w = self.edge_w + scale * priv
            self._edge_updates_seen += 1

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
            return {
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

    def accrue_counters(self) -> None:
        """Advance the running victim/loss totals one round (federated vs siloed)."""
        with self._lock:
            fed, silo = self._det()
            self.cum["fed"] += AT_RISK * (1 - fed)
            self.cum["silo"] += AT_RISK * (1 - silo)
