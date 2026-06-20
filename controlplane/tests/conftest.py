"""Shared pytest fixtures: a fresh control plane + TestClient per test."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from controlplane import crypto
from controlplane.server import app as app_module
from controlplane.state import DIM, ControlPlane

ADMIN_KEY = "test-admin-key"


@pytest.fixture
def plane():
    return ControlPlane(admin_key=ADMIN_KEY, min_updates=3)


@pytest.fixture
def client(plane):
    # Swap the module-level plane so the app dependencies see the fresh one.
    app_module.plane = plane
    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture
def admin_headers():
    return {"X-Admin-Key": ADMIN_KEY}


class FakeMember:
    """A fake bank node: keypair + helpers to enrol, approve, sign JWTs."""

    def __init__(self, client, member_id, admin_headers, approve=True):
        self.client = client
        self.member_id = member_id
        self.priv_pem, self.pub_pem = crypto.generate_keypair()
        r = client.post("/v1/members/enroll", json={
            "memberId": member_id,
            "displayName": f"{member_id} Bank",
            "publicKeyPem": self.pub_pem,
            "attestationQuote": f"quote-{member_id}",
        })
        assert r.status_code == 201, r.text
        self.tenant_id = r.json()["tenantId"]
        if approve:
            r2 = client.post(f"/v1/members/{member_id}/approve", headers=admin_headers)
            assert r2.status_code == 200, r2.text

    def jwt(self):
        return crypto.make_member_jwt(self.priv_pem, self.member_id, self.tenant_id)

    def auth_headers(self):
        return {"Authorization": f"Bearer {self.jwt()}"}

    def submit(self, round_, update, recall=0.8, num_examples=1000,
               silo_recall=None):
        local_metrics = {"recall": recall}
        if silo_recall is not None:
            local_metrics["siloRecall"] = silo_recall
        return self.client.post(
            f"/v1/rounds/{round_}/updates",
            headers=self.auth_headers(),
            json={
                "memberId": self.member_id,
                "round": round_,
                "update": [float(x) for x in update],
                "numExamples": num_examples,
                "localMetrics": local_metrics,
                "attestationQuote": f"quote-{self.member_id}",
            },
        )


@pytest.fixture
def make_member(client, admin_headers):
    def _make(member_id, approve=True):
        return FakeMember(client, member_id, admin_headers, approve=approve)
    return _make


@pytest.fixture
def honest_delta():
    """A small honest update vector of the right dimension."""
    def _delta(seed=0, scale=0.1):
        rng = np.random.default_rng(seed)
        return list(rng.normal(0, scale, size=DIM))
    return _delta
