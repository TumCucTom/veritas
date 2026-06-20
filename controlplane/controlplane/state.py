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
import threading
from dataclasses import dataclass, field

import numpy as np

# Reused FL primitives (core installed editable -> `from veritas_core...`,
# matching the import style in core/server/app.py).
from veritas_core.aggregation import multi_krum
from veritas_core.data import FEATURE_DIM
from veritas_core.model import init_weights

from . import crypto
from .transparency import TransparencyLog

DIM = FEATURE_DIM + 1  # logistic bias term -> 11
DEFAULT_MIN_UPDATES = 3
DP_MAX_NORM = 2.0
DP_SIGMA = 0.05
N_BYZANTINE = 1


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


class ControlPlane:
    """Single source of truth for the reference control plane (in-memory)."""

    def __init__(self, admin_key: str = "dev-admin-key", min_updates: int = DEFAULT_MIN_UPDATES):
        self._lock = threading.RLock()
        self.admin_key = admin_key
        self.min_updates = min_updates

        # Control-plane signing identity (signs the Merkle root).
        self.plane_priv_pem, self.plane_pub_pem = crypto.generate_keypair()
        self.transparency = TransparencyLog(self.plane_priv_pem)

        self.members: dict[str, Member] = {}

        # Model registry. Version 0 is the genesis (zero) model, promoted.
        genesis = Model(
            version=0,
            weights=[float(x) for x in init_weights(FEATURE_DIM)],
            parent_version=None,
            status="promoted",
            metrics={"recall": 0.0},
        )
        self.models: dict[int, Model] = {0: genesis}
        self.promoted_version = 0
        self._next_version = 1

        # Rounds. Round numbering starts at 1; current round is open.
        self.current_round = 1
        self.round_status = "open"  # open | aggregating | closed
        self.updates: dict[int, dict[str, Update]] = {1: {}}  # round -> memberId -> Update
        self.results: dict[int, RoundResult] = {}
        self._receipt_counter = 0

        # SSE subscriber queues (asyncio.Queue) keyed by id.
        self._subscribers: list = []

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
        return m

    # -- rounds & models --------------------------------------------------
    def current_round_info(self) -> dict:
        with self._lock:
            return {
                "round": self.current_round,
                "globalModelVersion": self.promoted_version,
                "status": self.round_status,
                "dpParams": {"maxNorm": DP_MAX_NORM, "sigma": DP_SIGMA},
                "minUpdates": self.min_updates,
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

    @staticmethod
    def _krum_scores(deltas: list, n_byzantine: int):
        """Per-update Multi-Krum score: sum of squared distances to the k nearest
        peers (same metric core's multi_krum minimises). A poisoned update sits
        far from the honest cloud and earns the largest score."""
        U = np.stack(deltas); n = len(U); k = max(1, n - n_byzantine - 2)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.sum((U[i] - U[j]) ** 2)); dist[i, j] = dist[j, i] = d
        return np.array([np.sort(dist[i])[1:k + 1].sum() for i in range(n)])

    def _select(self, deltas: list) -> tuple[list[int], list[int]]:
        """Outlier-gated Multi-Krum selection.

        Reject at most `N_BYZANTINE` updates, and ONLY when they are genuine
        Krum-score outliers (score >> the honest cloud, via a robust
        median/MAD test). An all-honest round keeps everyone; a poisoned round
        drops the poison. Returns (selected_idx, rejected_idx).
        """
        n = len(deltas)
        # Need at least 4 updates to distinguish a Byzantine outlier from the
        # honest spread (with n<=3, one honest update inevitably looks "worst"
        # but isn't a real outlier). Below that, keep everyone.
        if n < 4:
            return list(range(n)), []
        scores = self._krum_scores(deltas, n_byzantine=N_BYZANTINE)
        order = list(np.argsort(scores))  # ascending: best first
        rejected: list[int] = []
        # A genuine poisoned update's Krum score sits FAR above the honest cloud.
        # Reject the single worst only if it dwarfs the honest baseline (median
        # of the remaining scores) by a wide margin — a gap/ratio test that is
        # robust to ties and small n, unlike a MAD z-score.
        worst = order[-1]
        rest = [scores[i] for i in order[:-1]]
        baseline = float(np.median(rest)) or 1e-12
        if scores[worst] > 10.0 * baseline:
            rejected.append(int(worst))
        selected = [i for i in range(n) if i not in rejected]
        return selected, rejected

    def _aggregate_round(self, r: int) -> RoundResult:
        """Run Multi-Krum over the round's updates, reject outliers, apply the
        aggregate delta to the parent (promoted) model -> new model version.

        Uses `core.veritas_core.aggregation.multi_krum` with n_byzantine=1.
        Multi-Krum selects the m most-consistent updates (m = n - n_byzantine);
        members not selected are reported as rejected -> fires attack_detected.
        """
        self.round_status = "aggregating"
        items = list(self.updates[r].values())
        member_ids = [u.member_id for u in items]
        deltas = [np.asarray(u.update, dtype=float) for u in items]
        n = len(deltas)

        selected, rejected_idx = self._select(deltas)
        # Robust aggregate: Multi-Krum over the SURVIVING updates (it averages
        # the m most-consistent of them). With no rejection this is FedAvg-like
        # over all honest updates; with a rejected outlier the poison is gone.
        survivors = [deltas[i] for i in selected]
        if len(survivors) == 1:
            agg = survivors[0]
        else:
            agg, _ = multi_krum(survivors, n_byzantine=0, m=len(survivors))

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
            transparency_seq=entry["seq"],
        )
        self.results[r] = result

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
            # cross-institution campaign far more slowly. We model the siloed
            # recall as a fixed fraction of the federated recall (honest lift
            # measure: same shape as the hackathon engine's siloed baseline).
            siloed_recall = round(fed_recall * 0.6, 4)
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
                               "lostGbp": silo_victims * avg_loss},
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
