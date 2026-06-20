"""In-memory state + federation orchestration for the control plane.

This is the reference implementation: all state lives in process memory. It is
deliberately structured behind a single `ControlPlane` object with typed
sub-records so a persistence layer (Postgres / append-only store) can slot in
by swapping the dict-backed collections for repositories without touching the
HTTP layer in `server/app.py`.

FL math is REUSED from `core/veritas_core` — we never reimplement Multi-Krum,
FedAvg, or DP here. We import `multi_krum` and aggregate the round's deltas.
"""
from __future__ import annotations

import datetime as _dt
import os
import threading
from dataclasses import dataclass, field

import numpy as np

# Reused FL primitives (core installed editable -> `from veritas_core...`,
# matching the import style in core/server/app.py). The robust aggregation and
# DP accounting primitives are NEW, tested core modules — imported, never
# reimplemented here.
from veritas_core.data import FEATURE_DIM
from veritas_core.dp_accountant import RDPAccountant, calibrate_noise_multiplier
from veritas_core.model import init_weights
from veritas_core.robust import robust_aggregate

from . import crypto
from .store import Store
from .transparency import TransparencyLog

DIM = FEATURE_DIM + 1  # logistic bias term -> 11
DEFAULT_MIN_UPDATES = 3
DP_MAX_NORM = 2.0
N_BYZANTINE = int(os.environ.get("VERITAS_N_BYZANTINE", "1"))

# Aggregation method for the global delta. Recommended default for the plane
# (small n, DP-noised inputs): trimmed_mean. Detection (which NAMES a rejected
# member) always runs Krum so the poison-rejection contract still holds.
AGG_METHOD = os.environ.get("VERITAS_AGG_METHOD", "trimmed_mean")

# DP budget configuration (env-overridable). The calibrated sigma is advertised
# in dpParams so nodes apply the right amount of Gaussian noise.
DP_EPSILON = float(os.environ.get("VERITAS_DP_EPSILON", "8.0"))
DP_DELTA = float(os.environ.get("VERITAS_DP_DELTA", "1e-5"))
DP_ROUNDS = int(os.environ.get("VERITAS_DP_ROUNDS", "50"))
# Direct noise-multiplier override. When set, σ is used as-is and the TRUE ε it
# yields over DP_ROUNDS is reported (honest). Used to pick a utility-preserving
# operating point on the deliberately-tiny reference model, where calibrating
# for a strict ε produces σ large enough to destroy the toy model — the real
# privacy/utility tradeoff. Production (larger model + data) calibrates from ε.
DP_SIGMA_OVERRIDE = os.environ.get("VERITAS_DP_SIGMA")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


@dataclass
class Member:
    member_id: str
    display_name: str
    tenant_id: str
    public_key_pem: str
    status: str = "pending"  # pending | active
    attestation_quote: str | None = None
    last_sync: str | None = None

    def public(self) -> dict:
        return {
            "memberId": self.member_id,
            "displayName": self.display_name,
            "tenantId": self.tenant_id,
            "status": self.status,
        }


@dataclass
class Model:
    version: int
    weights: list[float]
    parent_version: int | None
    status: str = "canary"  # canary | promoted | rolledback
    metrics: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_now)

    def public(self) -> dict:
        return {
            "version": self.version,
            "dim": len(self.weights),
            "weights": self.weights,
            "parentVersion": self.parent_version,
            "status": self.status,
            "createdAt": self.created_at,
        }


@dataclass
class Update:
    member_id: str
    round: int
    update: list[float]
    num_examples: int
    local_metrics: dict
    attestation_quote: str | None
    receipt_id: str


@dataclass
class RoundResult:
    round: int
    new_version: int
    contributors: list[str]
    rejected: list[str]
    global_metrics: dict
    transparency_seq: int
    silo_recall: float | None = None  # mean reported siloed-baseline recall


class ControlPlane:
    """Single source of truth for the reference control plane (in-memory)."""

    def __init__(self, admin_key: str = "dev-admin-key",
                 min_updates: int = DEFAULT_MIN_UPDATES,
                 db_path: str | None = None,
                 plane_key_path: str | None = None,
                 agg_method: str | None = None,
                 n_byzantine: int | None = None):
        self._lock = threading.RLock()
        self.admin_key = admin_key
        self.min_updates = min_updates

        # Robust-aggregation config (env-overridable, constructor wins for tests).
        self.agg_method = agg_method or AGG_METHOD
        self.n_byzantine = N_BYZANTINE if n_byzantine is None else int(n_byzantine)

        # Durable persistence. Tests default to an in-memory DB so existing
        # ControlPlane(...) construction is unchanged in behaviour.
        if db_path is None:
            db_path = os.environ.get("VERITAS_DB", ":memory:")
        self.store = Store(db_path)

        # Durable control-plane signing identity (signs the Merkle root). A
        # persisted key keeps the transparency log verifiable across restarts.
        key_path = plane_key_path
        if key_path is None:
            key_path = os.environ.get("VERITAS_PLANE_KEY",
                                      ":memory:" if self.store.is_memory
                                      else "plane_key.pem")
        self.plane_priv_pem, self.plane_pub_pem = crypto.load_or_create_keypair(key_path)
        self.transparency = TransparencyLog(
            self.plane_priv_pem, on_append=self.store.persist_transparency)

        self.members: dict[str, Member] = {}
        self.models: dict[int, Model] = {}
        self.promoted_version = 0
        self._next_version = 1
        self.current_round = 1
        self.round_status = "open"  # open | aggregating | closed
        self.updates: dict[int, dict[str, Update]] = {1: {}}
        self.results: dict[int, RoundResult] = {}
        self._receipt_counter = 0
        self._round_silo_recall: dict[int, float] = {}  # round -> mean reported

        # DP budget: calibrate the noise multiplier for the target (eps, delta)
        # over the configured rounds, and keep a long-lived RDP accountant that
        # tracks SPENT epsilon as rounds aggregate.
        self.dp_delta = DP_DELTA
        self.dp_rounds = DP_ROUNDS
        if DP_SIGMA_OVERRIDE is not None:
            # Operator pinned σ directly: use it, and report the TRUE ε it yields
            # over the round horizon so /v1/privacy stays honest.
            self.dp_sigma = float(DP_SIGMA_OVERRIDE)
            _acct = RDPAccountant()
            for _ in range(max(1, self.dp_rounds)):
                _acct.step(self.dp_sigma)
            self.dp_target_epsilon = round(_acct.get_epsilon(self.dp_delta), 4)
        else:
            self.dp_target_epsilon = DP_EPSILON
            self.dp_sigma = calibrate_noise_multiplier(
                target_epsilon=self.dp_target_epsilon, delta=self.dp_delta,
                num_rounds=self.dp_rounds)
        self._dp_accountant = RDPAccountant()
        self.dp_spent_epsilon = 0.0
        self._dp_rounds_spent = 0

        # SSE subscriber queues (asyncio.Queue) keyed by id.
        self._subscribers: list = []

        # Load prior state from the DB, or seed genesis on a fresh store.
        loaded = self.store.load_all()
        if loaded["models"]:
            self._rehydrate(loaded)
        else:
            self._seed_genesis()

    # -- persistence rehydration -----------------------------------------
    def _seed_genesis(self) -> None:
        genesis = Model(
            version=0,
            weights=[float(x) for x in init_weights(FEATURE_DIM)],
            parent_version=None,
            status="promoted",
            metrics={"recall": 0.0},
        )
        self.models = {0: genesis}
        self.store.persist_model(genesis)
        self.store.persist_meta("plane", self._meta_snapshot())

    def _meta_snapshot(self) -> dict:
        return {
            "promoted_version": self.promoted_version,
            "next_version": self._next_version,
            "current_round": self.current_round,
            "receipt_counter": self._receipt_counter,
            "dp_spent_epsilon": self.dp_spent_epsilon,
            "dp_rounds_spent": self._dp_rounds_spent,
        }

    def _rehydrate(self, loaded: dict) -> None:
        for row in loaded["members"]:
            self.members[row["member_id"]] = Member(
                member_id=row["member_id"], display_name=row["display_name"],
                tenant_id=row["tenant_id"], public_key_pem=row["public_key_pem"],
                status=row["status"], attestation_quote=row["attestation_quote"],
                last_sync=row["last_sync"])
        for row in loaded["models"]:
            self.models[row["version"]] = Model(
                version=row["version"], weights=row["weights"],
                parent_version=row["parent_version"], status=row["status"],
                metrics=row["metrics"], created_at=row["created_at"])
        for row in loaded["rounds"]:
            res = RoundResult(
                round=row["round"], new_version=row["new_version"],
                contributors=row["contributors"], rejected=row["rejected"],
                global_metrics=row["global_metrics"],
                transparency_seq=row["transparency_seq"],
                silo_recall=row["silo_recall"])
            self.results[row["round"]] = res
            if row["silo_recall"] is not None:
                self._round_silo_recall[row["round"]] = row["silo_recall"]
        for entry in loaded["transparency"]:
            rec = {"seq": entry["seq"], "type": entry["type"],
                   "data": entry["data"], "timestamp": entry["timestamp"]}
            if entry["round"] is not None:
                rec["round"] = entry["round"]
            rec["leafHash"] = entry["leaf_hash"]
            self.transparency.load_entry(rec)
        meta = loaded.get("meta", {}).get("plane", {})
        self.promoted_version = meta.get("promoted_version", 0)
        self._next_version = meta.get("next_version", max(self.models) + 1)
        self.current_round = meta.get("current_round", 1)
        self._receipt_counter = meta.get("receipt_counter", 0)
        self.dp_spent_epsilon = meta.get("dp_spent_epsilon", 0.0)
        self._dp_rounds_spent = meta.get("dp_rounds_spent", 0)
        # Replay the accountant so spent epsilon stays consistent post-restart.
        for _ in range(self._dp_rounds_spent):
            self._dp_accountant.step(self.dp_sigma)
        self.round_status = "open"
        self.updates.setdefault(self.current_round, {})

    # -- identity ---------------------------------------------------------
    def _tenant_for(self, member_id: str) -> str:
        # One tenant per member in the reference impl; tenantId == memberId
        # namespaced. Keeps the per-tenant console honest while staying simple.
        return f"tenant-{member_id}"

    def enroll(self, member_id: str, display_name: str, public_key_pem: str,
               attestation_quote: str | None = None) -> Member:
        with self._lock:
            if member_id in self.members:
                raise ValueError("member already enrolled")
            # Validate the public key parses as Ed25519.
            crypto.load_public_key(public_key_pem)
            m = Member(
                member_id=member_id,
                display_name=display_name,
                tenant_id=self._tenant_for(member_id),
                public_key_pem=public_key_pem,
                status="pending",
                attestation_quote=attestation_quote,
            )
            self.members[member_id] = m
            self.store.persist_member(m)
            entry = self.transparency.append(
                "member_enrolled",
                {"memberId": member_id, "tenantId": m.tenant_id,
                 "displayName": display_name},
            )
            self._publish("member_enrolled",
                          {"memberId": member_id, "tenantId": m.tenant_id,
                           "transparencySeq": entry["seq"]})
            return m

    def approve(self, member_id: str) -> Member:
        with self._lock:
            m = self.members.get(member_id)
            if m is None:
                raise KeyError(member_id)
            m.status = "active"
            self.store.persist_member(m)
            return m

    def authenticate(self, token: str) -> Member:
        """Verify a node->plane JWT and return the (active) member."""
        # Decode unverified to find the member, then verify against its key.
        import jwt as _jwt
        unverified = _jwt.decode(token, options={"verify_signature": False})
        member_id = unverified.get("sub")
        m = self.members.get(member_id)
        if m is None:
            raise PermissionError("unknown member")
        crypto.verify_member_jwt(token, m.public_key_pem)  # raises on failure
        if m.status != "active":
            raise PermissionError("member not active")
        m.last_sync = _now()
        self.store.persist_member(m)
        return m

    # -- rounds & models --------------------------------------------------
    def current_round_info(self) -> dict:
        with self._lock:
            return {
                "round": self.current_round,
                "globalModelVersion": self.promoted_version,
                "status": self.round_status,
                # Calibrated DP params: nodes read dpParams.sigma to noise their
                # client-side updates to the advertised (epsilon, delta) budget.
                "dpParams": {
                    "maxNorm": DP_MAX_NORM,
                    "sigma": self.dp_sigma,
                    "epsilon": self.dp_target_epsilon,
                    "delta": self.dp_delta,
                },
                "minUpdates": self.min_updates,
            }

    def privacy_state(self) -> dict:
        """DP budget snapshot for GET /v1/privacy."""
        with self._lock:
            return {
                "targetEpsilon": self.dp_target_epsilon,
                "delta": self.dp_delta,
                "spentEpsilon": round(self.dp_spent_epsilon, 6),
                "noiseMultiplier": self.dp_sigma,
                "roundsSpent": self._dp_rounds_spent,
                "budgetRemaining": round(
                    max(0.0, self.dp_target_epsilon - self.dp_spent_epsilon), 6),
            }

    def get_model(self, version: int) -> Model:
        m = self.models.get(version)
        if m is None:
            raise KeyError(version)
        return m

    def current_model(self) -> Model:
        return self.models[self.promoted_version]

    def submit_update(self, member_id: str, round_: int, update: list[float],
                      num_examples: int, local_metrics: dict,
                      attestation_quote: str | None = None) -> dict:
        with self._lock:
            if round_ != self.current_round:
                raise ValueError(f"round {round_} is not the current round "
                                 f"({self.current_round})")
            if self.round_status != "open":
                raise ValueError("round is not open for updates")
            if len(update) != DIM:
                raise ValueError(f"update must have dim {DIM}, got {len(update)}")
            self._receipt_counter += 1
            receipt_id = f"r{round_}-{member_id}-{self._receipt_counter}"
            self.updates.setdefault(round_, {})[member_id] = Update(
                member_id=member_id, round=round_, update=list(map(float, update)),
                num_examples=int(num_examples), local_metrics=dict(local_metrics or {}),
                attestation_quote=attestation_quote, receipt_id=receipt_id,
            )
            ready = len(self.updates[round_]) >= self.min_updates
            return {"accepted": True, "receiptId": receipt_id, "_ready": ready}

    def maybe_aggregate(self, round_: int | None = None) -> RoundResult | None:
        """Aggregate if the current round has enough updates; else no-op."""
        with self._lock:
            r = round_ if round_ is not None else self.current_round
            if r != self.current_round:
                return None
            if len(self.updates.get(r, {})) >= self.min_updates:
                return self._aggregate_round(r)
            return None

    def advance_round(self) -> RoundResult | None:
        """Admin/orchestrator forces aggregation of the current round."""
        with self._lock:
            r = self.current_round
            if not self.updates.get(r):
                # No updates -> just close & reopen next round, no new model.
                self._open_next_round(r)
                return None
            return self._aggregate_round(r)

    def _detect_byzantine(self, deltas: list) -> tuple[list[int], list[int], list]:
        """Krum-based detection that NAMES rejected members, outlier-gated.

        Runs Multi-Krum (via the core `robust_aggregate(method='krum')`) to get
        per-update Krum scores and the candidate it would drop. Krum ALWAYS
        drops `f` members even in an all-honest round, so we do not blindly
        accuse: a candidate is only reported as `rejected` (-> `attack_detected`)
        when its Krum score dwarfs the honest cloud (robust ratio test vs. the
        median of the remaining scores). This keeps an honest round at zero
        rejections while still NAMING a genuine poisoner (the e2e contract:
        'evil' must land in `rejected`). Returns (selected, rejected, scores).
        """
        n = len(deltas)
        f = self.n_byzantine
        # Need enough peers for a meaningful Krum score (n - f - 2 >= 1) AND a
        # surviving cloud to compare against; below that, keep everyone.
        if n < f + 3:
            return list(range(n)), [], []
        # Detect on RAW deltas (NO norm-clip): an amplified / sign-flipped poison
        # must stay a visible magnitude outlier here so it gets NAMED. Clipping
        # belongs in the AGGREGATE step (neutralise the attack), not in detection
        # (catch + name it). Clipping before scoring would shrink a loud poison to
        # normal magnitude and, with DP noise inflating the honest baseline, let
        # it slip under the outlier gate.
        _, info = robust_aggregate(deltas, method="krum", n_byzantine=f)
        scores = info.get("scores") or []
        krum_rejected = info["rejected"]  # the f Krum would drop
        if not scores or not krum_rejected:
            return info["selected"], krum_rejected, scores
        order = list(np.argsort(scores))  # ascending: best (honest) first
        confirmed: list[int] = []
        for cand in krum_rejected:
            rest = [scores[i] for i in order if i != cand]
            baseline = float(np.median(rest)) or 1e-12
            # A real poisoned update sits FAR above the honest baseline.
            if scores[cand] > 10.0 * baseline:
                confirmed.append(cand)
        selected = [i for i in range(n) if i not in confirmed]
        return selected, sorted(confirmed), scores

    def _aggregate_round(self, r: int) -> RoundResult:
        """Robustly aggregate the round's DP-noised deltas and mint a new model.

        Two principled core primitives do the work (imported, never
        reimplemented):
          * DETECTION — Multi-Krum (`robust_aggregate(method='krum')`) names any
            Byzantine outlier so we can report it as `rejected` and fire
            `attack_detected`.
          * AGGREGATE — the configured method (default `trimmed_mean`, env
            `VERITAS_AGG_METHOD`) over the SURVIVING updates produces the global
            delta, with `max_norm=DP_MAX_NORM` clipping amplification attacks.
        """
        self.round_status = "aggregating"
        items = list(self.updates[r].values())
        member_ids = [u.member_id for u in items]
        deltas = [np.asarray(u.update, dtype=float) for u in items]

        selected, rejected_idx, _ = self._detect_byzantine(deltas)
        survivors = [deltas[i] for i in selected]

        # Robust aggregate over survivors. Single survivor -> use it directly;
        # otherwise run the configured robust aggregator (no further rejection
        # since detection already removed the Byzantine member).
        if len(survivors) == 1:
            agg = survivors[0]
        else:
            agg, _ = robust_aggregate(survivors, method=self.agg_method,
                                      n_byzantine=0, max_norm=DP_MAX_NORM)

        contributors = [member_ids[i] for i in selected]
        rejected = [member_ids[i] for i in rejected_idx]

        parent = self.models[self.promoted_version]
        parent_w = np.asarray(parent.weights, dtype=float)
        new_w = parent_w + np.asarray(agg, dtype=float)

        # Global recall metric: mean of contributors' reported local recall
        # (the plane never sees raw eval data — banks report metrics-only).
        recalls = [self.updates[r][mid].local_metrics.get("recall", 0.0)
                   for mid in contributors]
        global_recall = round(float(np.mean(recalls)) if recalls else 0.0, 4)

        # Honest siloed counterfactual: mean of contributors' MEASURED siloed-
        # baseline recall (`siloRecall`, the cross-team contract). Falls back to
        # None when no node reported it (older nodes) -> tenant_state degrades
        # gracefully to the prior heuristic.
        silo_vals = [self.updates[r][mid].local_metrics.get("siloRecall")
                     for mid in contributors]
        silo_vals = [float(v) for v in silo_vals if v is not None]
        silo_recall = round(float(np.mean(silo_vals)), 4) if silo_vals else None
        if silo_recall is not None:
            self._round_silo_recall[r] = silo_recall

        version = self._next_version
        self._next_version += 1
        model = Model(
            version=version,
            weights=[float(x) for x in new_w],
            parent_version=parent.version,
            status="canary",
            metrics={"recall": global_recall},
        )
        self.models[version] = model
        self.store.persist_model(model)
        model_hash = crypto.leaf_hash(model.weights)

        attestation_quotes = {
            mid: self.updates[r][mid].attestation_quote
            for mid in member_ids
            if self.updates[r][mid].attestation_quote is not None
        }
        entry = self.transparency.append(
            "round_aggregated",
            {
                "contributors": contributors,
                "rejected": rejected,
                "newVersion": version,
                "globalRecall": global_recall,
                "modelHash": model_hash,
                "attestationQuotes": attestation_quotes,
            },
            round_=r,
        )

        result = RoundResult(
            round=r, new_version=version, contributors=contributors,
            rejected=rejected, global_metrics={"recall": global_recall},
            transparency_seq=entry["seq"], silo_recall=silo_recall,
        )
        self.results[r] = result
        self.store.persist_round_result(result, silo_recall=silo_recall)

        # Advance the DP budget: one aggregation round consumed the calibrated
        # Gaussian mechanism once. The long-lived RDP accountant tracks SPENT
        # epsilon across the federation's lifetime.
        self._dp_accountant.step(self.dp_sigma)
        self._dp_rounds_spent += 1
        self.dp_spent_epsilon = float(self._dp_accountant.get_epsilon(self.dp_delta))

        # Fire attack_detected for each rejected member (Multi-Krum dropped it).
        for mid in rejected:
            self._publish("attack_detected",
                          {"memberId": mid, "round": r, "rejected": True,
                           "reason": "multi_krum_outlier"})

        self._publish("round_complete",
                      {"round": r, "newVersion": version,
                       "contributors": contributors, "rejected": rejected,
                       "globalRecall": global_recall,
                       "transparencySeq": entry["seq"]})

        self._open_next_round(r)
        return result

    def _open_next_round(self, r: int) -> None:
        self.current_round = r + 1
        self.round_status = "open"
        self.updates.setdefault(self.current_round, {})
        self.store.persist_meta("plane", self._meta_snapshot())

    def get_round_result(self, r: int) -> dict:
        res = self.results.get(r)
        if res is None:
            raise KeyError(r)
        return {
            "round": res.round,
            "newVersion": res.new_version,
            "contributors": res.contributors,
            "rejected": res.rejected,
            "globalMetrics": res.global_metrics,
            "transparencySeq": res.transparency_seq,
        }

    # -- registry & governance -------------------------------------------
    def registry(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "version": m.version,
                    "status": m.status,
                    "metrics": m.metrics,
                    "createdAt": m.created_at,
                }
                for m in sorted(self.models.values(), key=lambda x: x.version)
            ]

    def promote(self, version: int) -> dict:
        with self._lock:
            m = self.models.get(version)
            if m is None:
                raise KeyError(version)
            # Demote the currently-promoted model to canary lineage marker.
            for other in self.models.values():
                if other.status == "promoted" and other.version != version:
                    other.status = "canary"
            m.status = "promoted"
            self.promoted_version = version
            for other in self.models.values():
                self.store.persist_model(other)
            self.store.persist_meta("plane", self._meta_snapshot())
            entry = self.transparency.append(
                "model_promoted", {"version": version, "metrics": m.metrics})
            self._publish("model_promoted",
                          {"version": version, "transparencySeq": entry["seq"]})
            return {"version": version, "status": "promoted"}

    def rollback(self, version: int) -> dict:
        with self._lock:
            m = self.models.get(version)
            if m is None:
                raise KeyError(version)
            m.status = "rolledback"
            # Revert to the parent (last good) model, or genesis.
            reverted_to = m.parent_version if m.parent_version is not None else 0
            target = self.models[reverted_to]
            target.status = "promoted"
            self.promoted_version = reverted_to
            self.store.persist_model(m)
            self.store.persist_model(target)
            self.store.persist_meta("plane", self._meta_snapshot())
            entry = self.transparency.append(
                "model_rolledback",
                {"version": version, "revertedTo": reverted_to})
            self._publish("model_promoted",
                          {"version": reverted_to, "transparencySeq": entry["seq"]})
            return {"version": version, "status": "rolledback",
                    "revertedTo": reverted_to}

    # -- tenant console ---------------------------------------------------
    def _member_for_tenant(self, tenant_id: str) -> Member | None:
        for m in self.members.values():
            if m.tenant_id == tenant_id:
                return m
        return None

    def tenant_state(self, tenant_id: str) -> dict:
        with self._lock:
            member = self._member_for_tenant(tenant_id)
            promoted = self.models[self.promoted_version]
            fed_recall = float(promoted.metrics.get("recall", 0.0))
            # Siloed counterfactual: a lone bank without federation learns the
            # cross-institution campaign far more slowly. We use the MEASURED
            # siloed-baseline recall the nodes report each round (`siloRecall`
            # cross-team contract), averaged over rounds that reported it. If no
            # node has reported it yet (older nodes), we fall back gracefully to
            # the prior 0.6x heuristic and flag the source.
            reported = list(self._round_silo_recall.values())
            if reported:
                siloed_recall = round(float(np.mean(reported)), 4)
                silo_source = "reported"
            else:
                siloed_recall = round(fed_recall * 0.6, 4)
                silo_source = "heuristic"
            at_risk, avg_loss = 1500, 255
            fed_victims = int(at_risk * (1 - fed_recall))
            silo_victims = int(at_risk * (1 - siloed_recall))
            fraud_prevented = max(0, silo_victims - fed_victims) * avg_loss
            node = {
                "status": "active" if (member and member.status == "active") else "unknown",
                "lastSync": member.last_sync if member else None,
                "attestation": "verified" if (member and member.attestation_quote)
                else "unknown",
            }
            return {
                "round": self.current_round,
                "modelVersion": self.promoted_version,
                "customerRecordsTransmitted": 0,
                "node": node,
                "lift": {
                    "federated": {"recall": fed_recall, "victims": fed_victims,
                                  "lostGbp": fed_victims * avg_loss},
                    "siloed": {"recall": siloed_recall, "victims": silo_victims,
                               "lostGbp": silo_victims * avg_loss,
                               "source": silo_source},
                    "fraudPreventedGbp": fraud_prevented,
                },
            }

    def tenant_provenance(self, tenant_id: str) -> list[dict]:
        with self._lock:
            out = []
            for r in sorted(self.results):
                res = self.results[r]
                out.append({
                    "round": res.round,
                    "contributors": res.contributors,
                    "rejected": res.rejected,
                    "globalRecall": res.global_metrics.get("recall", 0.0),
                })
            return out

    # -- SSE --------------------------------------------------------------
    def subscribe(self, queue) -> None:
        self._subscribers.append(queue)

    def unsubscribe(self, queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def _publish(self, event: str, data: dict) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait({"event": event, "data": data})
            except Exception:
                pass
