"""An in-memory fake control plane implementing the PlaneTransport interface.

This lets the federation client be exercised end-to-end with NO network and NO
live control plane (which another agent owns). It models the Tier 2 contract
faithfully enough to integration-test the node:
  * enrolment registers the member's public key under a tenant,
  * authenticated update submission VERIFIES the EdDSA JWT against that key,
  * once minUpdates is reached it aggregates with Multi-Krum (reused from core)
    and promotes a new global model version.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from veritas_core.aggregation import multi_krum
from veritas_core import live_ensemble

from node.attestation import verify_quote
from node.identity import NodeIdentity


class FakePlane:
    def __init__(self, *, min_updates: int = 1, max_norm: float = 2.0, sigma: float = 0.05):
        # DIMENSION-AGNOSTIC plane: the expected wire dimension is DERIVED from
        # the genesis/global model length (the packed LiveEnsemble vector), not
        # a hardcoded number. Updates whose length differs are rejected. The
        # genesis is a real (seeded) live-ensemble vector, and ``self.dim`` is
        # read off its length so the two can never drift apart.
        self.global_w = live_ensemble.init_weights(seed=0)
        self.dim = self.global_w.shape[0]
        self.members: dict[str, dict] = {}        # memberId -> {publicKeyPem, tenantId, status}
        self.round = 1
        self.global_version = 0
        self.min_updates = min_updates
        self.max_norm = max_norm
        self.sigma = sigma
        self._round_updates: dict[int, list[np.ndarray]] = {}
        self.transparency: list[dict] = []
        self.enroll_count = 0
        self.update_count = 0
        self.last_update_body: dict | None = None  # captured for metric assertions
        self.last_enroll_body: dict | None = None   # captured for enrol-metadata assertions
        # ---- demo-control flags advertised in /v1/rounds/current ----------
        # The real plane piggybacks the demo RACE state on round-info; the fake
        # exposes the same fields so the node's reactions are testable offline.
        self.campaign_active = False
        self.attack_member_id: str | None = None
        self.epoch = 0

    # ---- transport surface (matches node.federation.transport.PlaneTransport) ----

    def enroll(self, body: dict[str, Any]) -> dict[str, Any]:
        member_id = body["memberId"]
        self.last_enroll_body = body
        tenant_id = f"tenant-{member_id}"
        public_key_pem = body["publicKeyPem"]
        # ATTESTATION: actually VERIFY the enrolment quote against the public key
        # being registered (binding the attestation to the enrolment identity),
        # instead of marking "verified" from mere presence. A forged or
        # self-asserted measurement is rejected here.
        quote = body.get("attestationQuote")
        if quote is not None:
            verify_quote(quote, enrolment_public_key_pem=public_key_pem,
                         expected_member_id=member_id)
        self.members[member_id] = {
            "publicKeyPem": public_key_pem,
            "tenantId": tenant_id,
            "status": "active",
            "attestationVerified": quote is not None,
        }
        self.enroll_count += 1
        self.transparency.append({"type": "member_enrolled", "memberId": member_id})
        return {"memberId": member_id, "tenantId": tenant_id, "status": "active"}

    def get_current_round(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "globalModelVersion": self.global_version,
            "status": "open",
            "dpParams": {"maxNorm": self.max_norm, "sigma": self.sigma},
            "minUpdates": self.min_updates,
            # Demo-control flags (shared contract with the control plane).
            "campaignActive": self.campaign_active,
            "attackMemberId": self.attack_member_id,
            "epoch": self.epoch,
        }

    def get_current_model(self) -> dict[str, Any]:
        return {
            "version": self.global_version,
            "dim": self.dim,
            "weights": self.global_w.tolist(),
            "parentVersion": self.global_version - 1 if self.global_version else None,
            "status": "promoted",
            "createdAt": 0,
        }

    def submit_update(self, round_no: int, body: dict[str, Any], token: str) -> dict[str, Any]:
        member_id = body["memberId"]
        member = self.members.get(member_id)
        if member is None:
            raise PermissionError("unknown member")
        # Verify the EdDSA JWT against the registered public key (the real plane's job).
        claims = NodeIdentity.verify_jwt(token, member["publicKeyPem"])
        assert claims["sub"] == member_id, "JWT sub must match memberId"
        assert claims["tid"] == member["tenantId"], "JWT tid must match tenant"

        # Per-round attestation is verified against the ENROLMENT key (not the
        # key in the quote), so a forged round quote is rejected.
        quote = body.get("attestationQuote")
        if quote is not None:
            verify_quote(quote, enrolment_public_key_pem=member["publicKeyPem"],
                         expected_member_id=member_id)

        update = np.asarray(body["update"], dtype=np.float64)
        # DIMENSION-AGNOSTIC validation: reject updates whose length != the
        # current global model length (derived, not hardcoded). Also reject
        # non-finite values before they poison aggregation.
        expected = self.global_w.shape[0]
        if update.shape[0] != expected:
            raise ValueError(
                f"update dim {update.shape[0]} != global model dim {expected}")
        if not np.all(np.isfinite(update)):
            raise ValueError("update contains non-finite values")

        self.update_count += 1
        self.last_update_body = body
        self._round_updates.setdefault(round_no, []).append(update)
        if len(self._round_updates[round_no]) >= self.min_updates:
            self._aggregate(round_no)
        return {"accepted": True, "receiptId": f"r{round_no}-{self.update_count}"}

    def get_round_result(self, round_no: int) -> dict[str, Any]:
        return {
            "round": round_no,
            "newVersion": self.global_version,
            "contributors": list(self.members.keys()),
            "rejected": [],
            "globalMetrics": {"recall": 0.0},
            "transparencySeq": len(self.transparency),
        }

    # ---- aggregation -----------------------------------------------------

    def _aggregate(self, round_no: int) -> None:
        ups = self._round_updates[round_no]
        if len(ups) == 1:
            agg = ups[0]                       # single contributor: apply directly
        else:
            m = max(1, len(ups) - 1)
            agg, _sel = multi_krum(ups, n_byzantine=1, m=m)
        self.global_w = self.global_w + agg
        self.global_version += 1
        self.round += 1
        self.transparency.append({"type": "round_aggregated", "round": round_no, "newVersion": self.global_version})
