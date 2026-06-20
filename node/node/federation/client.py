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
from typing import Callable

import httpx
import numpy as np

from veritas_core.attack import poisoned_update
from veritas_core.dp import add_noise
from veritas_core import live_ensemble

from ..attestation import Attestor
from ..config import EPOCHS, LR, MAX_NORM, SIGMA, PER_BLOCK_NORM
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
    # Demo-control reactions taken THIS round (for tests / observability).
    campaign_injected: bool = False
    poisoned: bool = False
    epoch: int = 0
    reset: bool = False


class FederationClient:
    def __init__(
        self,
        identity: NodeIdentity,
        transport: PlaneTransport,
        attestor: Attestor,
        *,
        rng: np.random.Generator | None = None,
        display_name: str | None = None,
        customers: int | None = None,
        on_campaign: Callable[[int], None] | None = None,
        on_reset: Callable[[int], None] | None = None,
    ):
        self.identity = identity
        self.transport = transport
        self.attestor = attestor
        self.rng = rng or np.random.default_rng(0)
        self.enrolled = False
        self.status = "unenrolled"
        self.last_round_submitted: int | None = None

        # ---- enrol metadata (real bank name + customer count) -------------
        # So the plane's banks[] shows the real institution, not the opaque id.
        self.display_name = display_name
        self.customers = customers

        # ---- demo-control reaction hooks ----------------------------------
        # ``on_campaign(epoch)`` injects the campaign LOCALLY (the runtime wires
        # this to the engine, choosing seeing/blind). ``on_reset(epoch)`` resets
        # local engine state to genesis. The client owns epoch bookkeeping so it
        # injects at most ONCE per epoch and detects epoch bumps (resets).
        self._on_campaign = on_campaign
        self._on_reset = on_reset
        self.current_epoch = 0
        self._campaign_injected_epoch: int | None = None

    # ---- enrolment -------------------------------------------------------

    def enroll(self) -> dict:
        """Register this node's public key + attestation quote with the plane."""
        quote = self.attestor.attest()
        body = {
            "memberId": self.identity.member_id,
            # Real bank name (engine.name) + customer count (engine.customers) so
            # the plane's banks[] shows the live institution. Fall back to the
            # opaque memberId if the runtime didn't supply metadata.
            "displayName": self.display_name or self.identity.member_id,
            "publicKeyPem": self.identity.public_key_pem(),
            "attestationQuote": quote.to_wire(),
        }
        if self.customers is not None:
            body["customers"] = int(self.customers)
        resp = self.transport.enroll(body)
        # Plane may assign/override the tenant; honour it for subsequent JWTs.
        self.identity.tenant_id = resp.get("tenantId", self.identity.tenant_id)
        self.status = resp.get("status", "pending")
        self.enrolled = True
        return resp

    # ---- one round -------------------------------------------------------

    def run_round(
        self, data, y: np.ndarray, *, silo_recall: float | None = None
    ) -> RoundResult:
        if not self.enrolled:
            self.enroll()

        rnd = self.transport.get_current_round()
        round_no = int(rnd["round"])
        dp = rnd.get("dpParams", {}) or {}
        max_norm = float(dp.get("maxNorm", MAX_NORM))
        sigma = float(dp.get("sigma", SIGMA))

        # ---- demo-control flags advertised in round-info ------------------
        # The control plane piggybacks the demo RACE state on the EXISTING poll
        # response (no new push channel): campaignActive / attackMemberId / epoch.
        campaign_active = bool(rnd.get("campaignActive", False))
        attack_member_id = rnd.get("attackMemberId")
        epoch = int(rnd.get("epoch", 0))

        reset_done = False
        # EPOCH BUMP (reset): when the plane advances the epoch, reset local
        # engine state to genesis so the demo can be re-run from scratch. Clear
        # our per-epoch injection bookkeeping too.
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            self._campaign_injected_epoch = None
            if self._on_reset is not None:
                self._on_reset(epoch)
                reset_done = True

        campaign_injected_now = False
        # CAMPAIGN: when the flag flips true and we haven't injected for THIS
        # epoch yet, inject the campaign locally (the runtime decides seeing vs
        # blind). Tracked per-epoch so it only injects once per run.
        if campaign_active and self._campaign_injected_epoch != epoch:
            self._campaign_injected_epoch = epoch
            if self._on_campaign is not None:
                self._on_campaign(epoch)
                campaign_injected_now = True

        model = self.transport.get_current_model()
        global_w = np.asarray(model["weights"], dtype=np.float64)
        version_before = int(model.get("version", 0))

        # Local training on the packed LiveEnsemble, then the delta the plane
        # aggregates (a single 617-vector — the contract is unchanged, only the
        # dimension grew). DP is PER-BLOCK: a single global clip on the packed
        # delta would be dominated by the large embedding/MLP blocks and crush
        # the small logistic/meta blocks, so we clip EACH block to its own norm
        # budget (live_ensemble.clip_blocks) BEFORE adding Gaussian noise.
        local_w = live_ensemble.train_local(global_w, data, y, epochs=EPOCHS, lr=LR)
        update = local_w - global_w
        clipped = live_ensemble.clip_blocks(update, PER_BLOCK_NORM)
        # Reuse the core Gaussian mechanism. The noise scale is sigma * max_norm;
        # we drive it off the plane-advertised max_norm so the demo's weak-privacy
        # operating point (sigma~0.1) still preserves utility on every block.
        priv = add_noise(clipped, sigma, max_norm, self.rng)
        local_recall = live_ensemble.recall(local_w, data, y)

        # ATTACK: if the plane has designated ME as the attacker this round,
        # poison my (already DP-clipped) update — sign-flip + amplify (reused
        # from core) — so the plane's Multi-Krum rejects me. Drives the real
        # attack_detected beat in the live federation.
        poisoned = False
        if attack_member_id is not None and str(attack_member_id) == str(self.identity.member_id):
            priv = poisoned_update(priv)
            poisoned = True

        demo = dict(
            campaign_injected=campaign_injected_now,
            poisoned=poisoned,
            epoch=epoch,
            reset=reset_done,
        )

        if str(rnd.get("status", "open")) not in ("open",):
            return RoundResult(
                round=round_no, submitted=False, local_recall=local_recall,
                global_version_before=version_before, global_version_after=version_before,
                update_norm=float(np.linalg.norm(priv)), reason=f"round status {rnd.get('status')}",
                **demo,
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
                    **demo,
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
            **demo,
        )

    def pull_global(self) -> tuple[np.ndarray, int]:
        model = self.transport.get_current_model()
        return np.asarray(model["weights"], dtype=np.float64), int(model.get("version", 0))
