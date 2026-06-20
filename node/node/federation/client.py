"""Federation client: the node→plane loop.

One full round (``run_round``):
  1. GET /v1/rounds/current   — round number, status, DP params, minUpdates.
  2. GET /v1/models/current   — the global weights to start from.
  3. train_local on connector data (reuse core), update = local_w - global_w.
  4. DP: privatize(update, maxNorm, sigma)  (reuse core.dp).
  5. POST /v1/rounds/{round}/updates  with an EdDSA JWT bearer + attestation quote.
  6. pull the new global model back (best-effort; the plane may still be aggregating).

Enrolment (``enroll``) runs once: register the Ed25519 public key + attestation
quote under the tenant, receive memberId/tenantId/status. The node then mints
its own JWTs locally for authenticated calls.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx
import numpy as np

from veritas_core.dp import privatize
from veritas_core.model import recall, train_local

from ..attestation import Attestor
from ..config import EPOCHS, LR, MAX_NORM, SIGMA
from ..identity import NodeIdentity
from .transport import PlaneTransport


@dataclass
class RoundResult:
    round: int
    submitted: bool
    local_recall: float
    global_version_before: int
    global_version_after: int
    update_norm: float
    reason: str = ""


class FederationClient:
    def __init__(
        self,
        identity: NodeIdentity,
        transport: PlaneTransport,
        attestor: Attestor,
        *,
        rng: np.random.Generator | None = None,
    ):
        self.identity = identity
        self.transport = transport
        self.attestor = attestor
        self.rng = rng or np.random.default_rng(0)
        self.enrolled = False
        self.status = "unenrolled"
        self.last_round_submitted: int | None = None

    # ---- enrolment -------------------------------------------------------

    def enroll(self) -> dict:
        """Register this node's public key + attestation quote with the plane."""
        quote = self.attestor.attest()
        body = {
            "memberId": self.identity.member_id,
            "displayName": self.identity.member_id,
            "publicKeyPem": self.identity.public_key_pem(),
            "attestationQuote": quote.to_wire(),
        }
        resp = self.transport.enroll(body)
        # Plane may assign/override the tenant; honour it for subsequent JWTs.
        self.identity.tenant_id = resp.get("tenantId", self.identity.tenant_id)
        self.status = resp.get("status", "pending")
        self.enrolled = True
        return resp

    # ---- one round -------------------------------------------------------

    def run_round(
        self, X: np.ndarray, y: np.ndarray, *, silo_recall: float | None = None
    ) -> RoundResult:
        if not self.enrolled:
            self.enroll()

        rnd = self.transport.get_current_round()
        round_no = int(rnd["round"])
        dp = rnd.get("dpParams", {}) or {}
        max_norm = float(dp.get("maxNorm", MAX_NORM))
        sigma = float(dp.get("sigma", SIGMA))

        model = self.transport.get_current_model()
        global_w = np.asarray(model["weights"], dtype=np.float64)
        version_before = int(model.get("version", 0))

        # Local training (reuse core), then the delta the plane aggregates.
        local_w = train_local(global_w, X, y, epochs=EPOCHS, lr=LR)
        update = local_w - global_w
        priv = privatize(update, max_norm, sigma, self.rng)
        local_recall = recall(local_w, X, y)

        if str(rnd.get("status", "open")) not in ("open",):
            return RoundResult(
                round=round_no, submitted=False, local_recall=local_recall,
                global_version_before=version_before, global_version_after=version_before,
                update_norm=float(np.linalg.norm(priv)), reason=f"round status {rnd.get('status')}",
            )

        quote = self.attestor.attest()
        token = self.identity.mint_jwt()
        # Honest-metrics contract: report BOTH the federated recall and the
        # MEASURED siloed-baseline recall. The control plane consumes
        # ``siloRecall`` to replace its hardcoded counterfactual; the field name
        # is the agreed cross-team contract.
        local_metrics = {"recall": round(float(local_recall), 4)}
        if silo_recall is not None:
            local_metrics["siloRecall"] = round(float(silo_recall), 4)
        body = {
            "memberId": self.identity.member_id,
            "round": round_no,
            "update": priv.tolist(),
            "numExamples": int(len(y)),
            "localMetrics": local_metrics,
            "attestationQuote": quote.to_wire(),
        }
        try:
            self.transport.submit_update(round_no, body, token)
        except httpx.HTTPStatusError as exc:
            # A pending/unapproved member is rejected with 401/403 until an
            # admin approves it. This is expected during the enrolment→approval
            # window: don't crash the loop, just report not-submitted and retry
            # on the next round (the loop keeps polling until the member is
            # active). Re-raise any other error (plane down, 5xx, etc.).
            if exc.response is not None and exc.response.status_code in (401, 403):
                self.status = "pending"
                return RoundResult(
                    round=round_no, submitted=False, local_recall=local_recall,
                    global_version_before=version_before,
                    global_version_after=version_before,
                    update_norm=float(np.linalg.norm(priv)),
                    reason="awaiting approval (member not active)",
                )
            raise
        self.status = "active"
        self.last_round_submitted = round_no

        # Pull the (possibly) new global model back.
        after = self.transport.get_current_model()
        version_after = int(after.get("version", version_before))

        return RoundResult(
            round=round_no,
            submitted=True,
            local_recall=local_recall,
            global_version_before=version_before,
            global_version_after=version_after,
            update_norm=float(np.linalg.norm(priv)),
        )

    def pull_global(self) -> tuple[np.ndarray, int]:
        model = self.transport.get_current_model()
        return np.asarray(model["weights"], dtype=np.float64), int(model.get("version", 0))
