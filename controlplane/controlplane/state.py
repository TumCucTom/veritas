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
from veritas_core import live_ensemble
from veritas_core.robust import robust_aggregate

from . import crypto
from .store import Store
from .transparency import TransparencyLog

# The LIVE federated/siloed model is now the stacked LiveEnsemble (packed as a
# single flat vector). The node<->plane contract is UNCHANGED — one weight
# vector per round — only its dimension grows 11 -> 617 (the packed ensemble).
DIM = live_ensemble.weight_dim()  # packed [logistic|mlp|embeddings|meta] -> 617
DEFAULT_MIN_UPDATES = 3

# Named feature contract (mirrors the core FEATURE_DIM contract + the node's
# FEATURE_ORDER). /predict builds the scored vector from the REAL transaction and
# derives indicators from the actual per-feature contributions to the score.
FEATURE_ORDER = [
    "amount", "oldOrig", "newOrig", "oldDest", "newDest",
    "accountAge", "velocity", "fanout", "isTransfer", "campaignSig",
]
assert len(FEATURE_ORDER) == FEATURE_DIM, "FEATURE_ORDER must match FEATURE_DIM"
# Human-readable fraud indicator per feature, surfaced when that feature pushes
# the score TOWARD fraud (i.e. its signed contribution x[i]*w[i] is positive).
FEATURE_INDICATORS = {
    "amount": "anomalous transfer amount",
    "accountAge": "new account",
    "velocity": "high transaction velocity",
    "fanout": "fan-out to multiple recipients",
    "isTransfer": "transfer-type movement",
    "campaignSig": "matches active campaign signature",
    "oldOrig": "originator balance anomaly",
    "newOrig": "originator balance drained",
    "oldDest": "recipient balance anomaly",
    "newDest": "recipient balance spike",
}

# Legacy single-backend demo counter model — REPLICATED from
# core/veritas_core/engine.py so the control plane serves the exact shape the
# web's useVeritas store consumes, but driven by REAL member-reported metrics.
DEMO_AVG_LOSS = 255      # engine.AVG_LOSS
DEMO_AT_RISK = 1500      # engine.AT_RISK
DEMO_THRESH = 0.9        # engine.THRESH
DEMO_HOURS = 1.0         # engine.HOURS
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
    # Demo metadata (legacy single-backend contract): customer count surfaced as
    # banks[].customers, and the most recent reported per-bank detection metrics.
    customers: int = 0
    last_metrics: dict = field(default_factory=dict)

    def public(self) -> dict:
        return {
            "memberId": self.member_id,
            "displayName": self.display_name,
            "tenantId": self.tenant_id,
            "status": self.status,
            "customers": self.customers,
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

        # SSE subscriber queues (asyncio.Queue) keyed by id. The running event
        # loop is captured at server startup so worker-thread publishes are
        # marshalled back onto it thread-safely (see set_event_loop / _publish).
        self._subscribers: list = []
        self._event_loop = None

        # Legacy single-backend demo state. campaign/attack flags drive the
        # banks[] poisoned markers + counters; epoch bumps on /sim/reset so nodes
        # re-genesis. Cumulative victim/loss totals accrue per aggregated round
        # from the MEAN reported detection across active banks (engine.cum).
        self.campaign_active = False
        self.attack_active = False
        self.attack_member_id: str | None = None
        self.epoch = 0
        self._demo_cum = {"fed": 0.0, "silo": 0.0}

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
            weights=[float(x) for x in live_ensemble.init_weights()],
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
                last_sync=row["last_sync"],
                customers=int(row.get("customers") or 0))
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
               attestation_quote: str | None = None,
               customers: int = 0) -> Member:
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
                customers=int(customers or 0),
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
                # Legacy demo flags so nodes react via their poll loop (NO push):
                # nodes inject the campaign / poison their own update / re-genesis
                # on epoch change based on these.
                "campaignActive": self.campaign_active,
                "attackMemberId": self.attack_member_id,
                "epoch": self.epoch,
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

    def expected_update_dim(self) -> int:
        """Expected length of a member update vector, derived from the CURRENT
        global model rather than a hardcoded constant. Updates are deltas applied
        to the promoted model, so they MUST match its weight-vector length. This
        keeps the plane dimension-agnostic: swap in a longer live-ensemble model
        and the accepted update dimension follows automatically (no magic-number
        change on the wire)."""
        return len(self.current_model().weights)

    def submit_update(self, member_id: str, round_: int, update: list[float],
                      num_examples: int, local_metrics: dict,
                      attestation_quote: str | None = None) -> dict:
        with self._lock:
            if round_ != self.current_round:
                raise ValueError(f"round {round_} is not the current round "
                                 f"({self.current_round})")
            if self.round_status != "open":
                raise ValueError("round is not open for updates")
            # Dimension-agnostic: the expected weight dimension is whatever the
            # CURRENT global model carries (genesis -> FEATURE_DIM+1, but a live
            # ensemble model can be any length). Never hardcode a magic number —
            # derive it so nodes can send live-ensemble-dim updates without any
            # plane-side change.
            expected_dim = self.expected_update_dim()
            if len(update) != expected_dim:
                raise ValueError(
                    f"update must have dim {expected_dim} (current global model "
                    f"length), got {len(update)}")
            # Reject non-finite values (NaN / +-inf): a single poisoned NaN would
            # otherwise propagate through aggregation and corrupt the model.
            arr = np.asarray(update, dtype=float)
            if not np.all(np.isfinite(arr)):
                raise ValueError("update contains non-finite values (NaN/inf)")
            self._receipt_counter += 1
            receipt_id = f"r{round_}-{member_id}-{self._receipt_counter}"
            metrics = dict(local_metrics or {})
            self.updates.setdefault(round_, {})[member_id] = Update(
                member_id=member_id, round=round_, update=list(map(float, update)),
                num_examples=int(num_examples), local_metrics=metrics,
                attestation_quote=attestation_quote, receipt_id=receipt_id,
            )
            # Remember the member's latest reported detection metrics so the
            # legacy banks[] view shows the real per-bank federated/siloed recall.
            m = self.members.get(member_id)
            if m is not None:
                m.last_metrics = metrics
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
        """Aggregate round `r`, restoring an OPEN round on any failure.

        The aggregation transition (`open -> aggregating -> open`) must be crash-
        safe: if any step inside raises (a malformed update slipping past
        validation, a core-aggregator error, a persistence failure), the round
        would otherwise wedge on `aggregating` forever and reject every further
        update — a denial of service. We wrap the work so that on ANY exception
        we log it and restore `round_status = "open"` (the round is NOT advanced,
        so the same updates can be retried) before re-raising. On success the
        normal path advances to the next open round.
        """
        try:
            return self._aggregate_round_impl(r)
        except Exception as exc:  # noqa: BLE001 - never wedge on "aggregating"
            # Restore the round to OPEN so the federation isn't stuck. Keep the
            # same round number + accumulated updates so a retry is possible.
            if self.round_status == "aggregating" and self.current_round == r:
                self.round_status = "open"
            import logging
            logging.getLogger(__name__).exception(
                "round %s aggregation failed; restored status to 'open': %s",
                r, exc)
            raise

    def _aggregate_round_impl(self, r: int) -> RoundResult:
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
            # AGGREGATE-step clip is DISABLED (max_norm=None) for the packed
            # ensemble: a single global L2 clip on the 617-vector is dominated by
            # the large embedding/MLP blocks and would crush the small logistic /
            # meta blocks (the heterogeneous-scale problem live_ensemble.clip_blocks
            # solves). The node already applies PER-BLOCK DP clipping client-side
            # before noising, so amplification attacks are bounded per block there;
            # detection (Krum on RAW deltas, below) still NAMES a loud poisoner.
            agg, _ = robust_aggregate(survivors, method=self.agg_method,
                                      n_byzantine=0, max_norm=None)

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

        # Legacy demo accrual + SSE: increment cumulative victim/loss counters
        # off the mean reported detection across banks, and emit the legacy
        # round_complete / client_updated / attack_detected stream the web store
        # consumes. Done before opening the next round so banks[] reflect THIS
        # round's reported metrics.
        demo_banks = self._demo_banks()
        self._accrue_demo_counters(demo_banks)
        self._publish_demo_round(demo_banks, rejected)

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

    # -- legacy single-backend demo contract ------------------------------
    # The web's useVeritas store consumes the original single-backend shape.
    # These methods serve that EXACT shape, backed by aggregating the real
    # enrolled federation members (per-bank detection from each member's latest
    # reported localMetrics; counters REPLICATED from engine.py off the mean).
    def _active_members(self) -> list[Member]:
        return [m for m in self.members.values() if m.status == "active"]

    def _latest_rejected(self) -> list[str]:
        """Members rejected in the most recent aggregated round."""
        if not self.results:
            return []
        last = self.results[max(self.results)]
        return list(last.rejected)

    def _demo_banks(self) -> list[dict]:
        rejected = set(self._latest_rejected())
        banks = []
        for m in sorted(self._active_members(), key=lambda x: x.member_id):
            metrics = m.last_metrics or {}
            fed = float(metrics.get("recall", 0.0) or 0.0)
            silo = float(metrics.get("siloRecall", 0.0) or 0.0)
            banks.append({
                "id": m.member_id,
                "name": m.display_name or m.member_id,
                "customers": int(m.customers or 0),
                "detection": {"federated": round(fed, 3), "siloed": round(silo, 3)},
                "poisoned": m.member_id in rejected,
            })
        return banks

    def _accrue_demo_counters(self, banks: list[dict]) -> None:
        """Increment cumulative victim/loss totals from the MEAN detection across
        active banks — the control-plane analogue of engine.cum, accrued once per
        aggregated round."""
        if not banks:
            return
        n = len(banks)
        fed_mean = sum(b["detection"]["federated"] for b in banks) / n
        silo_mean = sum(b["detection"]["siloed"] for b in banks) / n
        self._demo_cum["fed"] += DEMO_AT_RISK * (1 - fed_mean)
        self._demo_cum["silo"] += DEMO_AT_RISK * (1 - silo_mean)

    def _demo_counters(self, banks: list[dict]) -> dict:
        n = max(1, len(banks))
        fed_mean = sum(b["detection"]["federated"] for b in banks) / n
        silo_mean = sum(b["detection"]["siloed"] for b in banks) / n
        fv = int(self._demo_cum["fed"])
        sv = int(self._demo_cum["silo"])

        def ttd(mean: float) -> float:
            return DEMO_HOURS * self.current_round if mean >= DEMO_THRESH else 101.0

        return {
            "federated": {
                "fraudPreventedGbp": max(0, sv - fv) * DEMO_AVG_LOSS,
                "timeToDetectHours": ttd(fed_mean),
                "victims": fv,
                "lostGbp": fv * DEMO_AVG_LOSS,
            },
            "siloed": {
                "fraudPreventedGbp": 0,
                "timeToDetectHours": ttd(silo_mean),
                "victims": sv,
                "lostGbp": sv * DEMO_AVG_LOSS,
            },
        }

    def demo_state(self, gnn_benchmark: dict | None = None) -> dict:
        """The legacy single-backend State shape (GET /state)."""
        # Fall back to the benchmark stored on the plane so SSE round_complete
        # payloads (published from inside aggregation, with no caller arg) carry
        # gnnBenchmark too — otherwise the web's store, overwritten by each round
        # event, would blank the GNN panel after the first round.
        if gnn_benchmark is None:
            gnn_benchmark = getattr(self, "gnn_benchmark", None)
        with self._lock:
            banks = self._demo_banks()
            state = {
                "round": self.current_round,
                "running": True,
                "banks": banks,
                "counters": self._demo_counters(banks),
                "campaignActive": self.campaign_active,
                "attackActive": self.attack_active,
                "customerRecordsTransmitted": 0,
            }
            if gnn_benchmark is not None:
                b = dict(gnn_benchmark)
                from veritas_core.gnn_benchmark import benchmark_current
                b["current"] = benchmark_current(
                    gnn_benchmark, self.current_round, self.campaign_active)
                state["gnnBenchmark"] = b
            return state

    def demo_step(self) -> dict:
        """Advance ONE federation round for the demo: aggregate the current
        round's submitted updates (reuse _aggregate_round) and auto-promote the
        new canary so the served global model advances. Counter accrual + legacy
        SSE happen inside _aggregate_round. If no updates yet, still open the next
        round so nodes can submit. Returns the legacy State."""
        with self._lock:
            r = self.current_round
            if self.updates.get(r):
                result = self._aggregate_round(r)
                # Auto-promote the freshly-minted canary so /predict and banks[]
                # advance for the demo (no manual governance step needed).
                if result is not None:
                    self.promote(result.new_version)
            else:
                self._open_next_round(r)
            return self.demo_state()

    def demo_inject_campaign(self, typology: str | None = None) -> dict:
        with self._lock:
            self.campaign_active = True
            return {"ok": True}

    def demo_inject_attack(self, member_id: str | None) -> dict:
        with self._lock:
            self.attack_member_id = member_id
            self.attack_active = True
            return {"ok": True}

    def demo_reset(self) -> None:
        """Reset the demo: round=1, clear flags + cumulative counters, re-genesis
        the global model, and bump the epoch so nodes re-genesis too."""
        with self._lock:
            self.campaign_active = False
            self.attack_active = False
            self.attack_member_id = None
            self.epoch += 1
            self._demo_cum = {"fed": 0.0, "silo": 0.0}
            self.current_round = 1
            self.round_status = "open"
            self.updates = {1: {}}
            self.results = {}
            self._round_silo_recall = {}
            # Re-genesis the served global model so detection restarts from scratch.
            genesis = Model(
                version=0,
                weights=[float(x) for x in live_ensemble.init_weights()],
                parent_version=None,
                status="promoted",
                metrics={"recall": 0.0},
            )
            self.models = {0: genesis}
            self.promoted_version = 0
            self._next_version = 1
            for m in self.members.values():
                m.last_metrics = {}
            self.store.persist_model(genesis)
            self.store.persist_meta("plane", self._meta_snapshot())

    def _transaction_to_features(self, txn: dict) -> np.ndarray:
        """Build a FEATURE_DIM vector from the ACTUAL request transaction.

        Named features pass through directly; if the caller only signals a
        campaign match we synthesise the "safe-account mule" shape the federated
        model learned (benign generic axes, strong campaign signature) — mirrors
        the node's `transaction_to_features` so the served model and the trained
        model speak the same feature contract."""
        x = np.zeros(FEATURE_DIM, dtype=np.float64)
        for idx, name in enumerate(FEATURE_ORDER):
            if name in txn:
                try:
                    x[idx] = float(txn[name])
                except (TypeError, ValueError):
                    pass
        if "campaignSignature" in txn or "campaignSig" in txn:
            sig = float(txn.get("campaignSignature", txn.get("campaignSig", 1.0)))
            x[FEATURE_ORDER.index("campaignSig")] = 1.5 * sig
            x[FEATURE_ORDER.index("amount")] = -0.5
            x[FEATURE_ORDER.index("velocity")] = -0.5
            x[FEATURE_ORDER.index("fanout")] = -0.5
            x[FEATURE_ORDER.index("accountAge")] = -2.0
        return x

    def demo_predict(self, transaction: dict | None) -> dict:
        """Score a transaction with the plane's CURRENT global model (the packed
        LiveEnsemble) and derive the indicators from the REAL per-feature
        contributions to that score.

        The scored vector is built from the actual request transaction (not a
        fixed mule row), and each indicator corresponds to a feature whose signed
        contribution x[i]*w[i] pushes the score toward fraud — so two different
        transactions yield different, faithful indicator lists. Response shape is
        a superset of the legacy contract (`label`/`confidence`/`indicators`)
        plus the {score, indicators, regime} fields."""
        with self._lock:
            txn = transaction or {}
            x_row = self._transaction_to_features(txn)
            w = np.asarray(self.current_model().weights, dtype=float)
            # Score with the packed ensemble's own scorer: a bare FEATURE_DIM row
            # is scored with a neutral categorical view (the campaign typology is
            # tabular). predict_one handles unpacking the 617-dim packed vector.
            p = float(live_ensemble.predict_one(w, x_row))

            # Per-feature signed contribution to the logit: x[i] * w[i]. Features
            # with a POSITIVE contribution are the ones actively driving the score
            # toward fraud — those become the indicators, ordered by magnitude.
            feat_w = w[:FEATURE_DIM]
            contributions = x_row * feat_w
            ranked = sorted(
                range(FEATURE_DIM), key=lambda i: contributions[i], reverse=True)
            indicators = [
                FEATURE_INDICATORS[FEATURE_ORDER[i]]
                for i in ranked
                if contributions[i] > 1e-9 and FEATURE_ORDER[i] in FEATURE_INDICATORS
            ]
            # Always surface at least the top contributing feature so the panel
            # never renders an empty list (e.g. an untrained genesis model where
            # all weights are zero falls back to the dominant feature axis).
            if not indicators:
                top = ranked[0]
                indicators = [FEATURE_INDICATORS.get(
                    FEATURE_ORDER[top], "model-flagged anomaly")]

            label = "fraud" if p >= 0.5 else "legitimate"
            return {
                "label": label,
                "confidence": round(p, 3),
                "score": round(p, 3),
                "regime": "federated",
                "indicators": indicators,
            }

    def _publish_demo_round(self, banks: list[dict], rejected: list[str]) -> None:
        """Emit the legacy SSE events the web store subscribes to (on /events)."""
        self._publish("round_complete", self.demo_state())
        for b in banks:
            self._publish("client_updated",
                          {"bankId": b["id"], "detection": b["detection"]})
        for mid in rejected:
            self._publish("attack_detected", {"bankId": mid, "rejected": True})

    # -- SSE --------------------------------------------------------------
    def set_event_loop(self, loop) -> None:
        """Capture the server's running event loop at startup.

        State-mutating handlers (enroll/aggregate/...) run in FastAPI's sync
        threadpool, i.e. on worker threads with NO running event loop. Pushing to
        an asyncio.Queue from there with a bare `put_nowait` is NOT thread-safe
        and can silently lose/corrupt events. With the loop captured, `_publish`
        schedules the enqueue back ONTO the loop via `call_soon_threadsafe` (the
        same pattern the node uses), so events are delivered reliably."""
        self._event_loop = loop

    def subscribe(self, queue) -> None:
        self._subscribers.append(queue)

    def unsubscribe(self, queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def _publish(self, event: str, data: dict) -> None:
        loop = getattr(self, "_event_loop", None)
        msg = {"event": event, "data": data}
        for q in list(self._subscribers):
            try:
                if loop is not None and not loop.is_closed():
                    # Thread-safe hand-off from a worker thread back to the loop
                    # that owns the asyncio.Queue. Never lose the event.
                    loop.call_soon_threadsafe(q.put_nowait, msg)
                else:
                    # No loop captured (unit tests drain a plain queue inline):
                    # a direct put_nowait is correct since there is no concurrent
                    # loop consuming the queue.
                    q.put_nowait(msg)
            except Exception:
                pass
