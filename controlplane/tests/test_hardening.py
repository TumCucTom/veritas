"""Control-plane hardening: input robustness, dimension-agnosticism, JWT
binding, crash-safe aggregation, constant-time admin auth, env-driven CORS, and
real per-feature /predict indicators."""
from __future__ import annotations

import numpy as np
import pytest

from controlplane import crypto
from controlplane.state import DIM, ControlPlane


def _enroll(plane, mids):
    for mid in mids:
        _, pub = crypto.generate_keypair()
        plane.enroll(mid, mid, pub)
        plane.approve(mid)


# --------------------------------------------------------------------------- #
# Input robustness: non-finite + wrong dimension.
# --------------------------------------------------------------------------- #
def test_reject_non_finite_update(plane):
    _enroll(plane, ["n0"])
    bad = [0.0] * DIM
    bad[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        plane.submit_update("n0", 1, bad, 1000, {"recall": 0.8})
    bad[0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        plane.submit_update("n0", 1, bad, 1000, {"recall": 0.8})


def test_reject_non_finite_update_over_http(client, make_member):
    m = make_member("n0")
    # A body literally containing the non-standard JSON `NaN` token (which the
    # stdlib parser accepts by default) must be rejected with a clear 4xx, not
    # silently aggregated.
    coords = ",".join(["0.0"] * (DIM - 1) + ["NaN"])
    body = ('{"memberId":"n0","round":1,"numExamples":1,"localMetrics":{},'
            f'"update":[{coords}]}}')
    headers = {**m.auth_headers(), "Content-Type": "application/json"}
    r = client.post("/v1/rounds/1/updates", headers=headers, content=body)
    assert r.status_code == 400
    assert "non-finite" in r.json()["detail"]


def test_dimension_derived_from_current_model_not_hardcoded(plane):
    """The accepted update length is whatever the CURRENT global model carries,
    derived — never a hardcoded 11."""
    _enroll(plane, ["n0"])
    assert plane.expected_update_dim() == len(plane.current_model().weights)
    wrong = [0.0] * (plane.expected_update_dim() + 3)
    with pytest.raises(ValueError, match="current global model length"):
        plane.submit_update("n0", 1, wrong, 1000, {"recall": 0.8})

    # If a longer model is promoted, the accepted dimension follows it with NO
    # code change — proving the plane is dimension-agnostic.
    from controlplane.state import Model
    longer = Model(version=99, weights=[0.0] * 32, parent_version=0,
                   status="promoted")
    plane.models[99] = longer
    plane.promoted_version = 99
    assert plane.expected_update_dim() == 32
    r = plane.submit_update("n0", 1, [0.01] * 32, 1000, {"recall": 0.8})
    assert r["accepted"] is True


def test_wrong_dimension_rejected_over_http(client, make_member):
    m = make_member("n0")
    r = client.post("/v1/rounds/1/updates", headers=m.auth_headers(), json={
        "memberId": "n0", "round": 1, "update": [0.0] * 5, "numExamples": 1,
        "localMetrics": {}})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# JWT hardening: aud/iss/jti issued + verified.
# --------------------------------------------------------------------------- #
def test_jwt_carries_and_verifies_aud_iss_jti():
    import jwt as _jwt
    priv, pub = crypto.generate_keypair()
    tok = crypto.make_member_jwt(priv, "m0", "tenant-m0")
    claims = _jwt.decode(tok, options={"verify_signature": False})
    assert claims["aud"] == crypto.JWT_AUDIENCE
    assert claims["iss"] == crypto.JWT_ISSUER
    assert claims["jti"]
    # Round-trip verification succeeds.
    assert crypto.verify_member_jwt(tok, pub)["sub"] == "m0"


def test_jwt_wrong_audience_rejected():
    import jwt as _jwt
    priv, pub = crypto.generate_keypair()
    tok = crypto.make_member_jwt(priv, "m0", "tenant-m0", audience="some-other-svc")
    with pytest.raises(_jwt.PyJWTError):
        crypto.verify_member_jwt(tok, pub)  # default audience mismatch


def test_jwt_ttl_is_short_by_default():
    import jwt as _jwt
    priv, _ = crypto.generate_keypair()
    tok = crypto.make_member_jwt(priv, "m0", "tenant-m0")
    claims = _jwt.decode(tok, options={"verify_signature": False})
    assert claims["exp"] - claims["iat"] == crypto.JWT_DEFAULT_TTL_SECONDS
    assert crypto.JWT_DEFAULT_TTL_SECONDS <= 600


# --------------------------------------------------------------------------- #
# Crash-safe aggregation: an exception restores an OPEN round (no DoS wedge).
# --------------------------------------------------------------------------- #
def test_aggregation_failure_restores_open_round(plane, monkeypatch):
    _enroll(plane, ["n0", "n1", "n2"])
    plane.min_updates = 3
    base = np.random.default_rng(1).normal(0, 0.02, size=DIM)
    for i, mid in enumerate(["n0", "n1", "n2"]):
        jit = np.random.default_rng(10 + i).normal(0, 0.004, size=DIM)
        plane.submit_update(mid, 1, list(base + jit), 1000, {"recall": 0.8})

    import controlplane.state as st

    def boom(*a, **k):
        raise RuntimeError("aggregator exploded")

    monkeypatch.setattr(st, "robust_aggregate", boom)
    with pytest.raises(RuntimeError):
        plane.advance_round()

    # NOT wedged on "aggregating": round is open again, same round, retryable.
    assert plane.round_status == "open"
    assert plane.current_round == 1

    monkeypatch.undo()
    # A retry now succeeds and advances normally.
    res = plane.advance_round()
    assert res is not None and plane.round_status == "open"
    assert plane.current_round == 2


# --------------------------------------------------------------------------- #
# Constant-time admin auth.
# --------------------------------------------------------------------------- #
def test_admin_auth_constant_time_and_rejects_wrong_key(client):
    # Wrong key rejected, missing key rejected.
    assert client.post("/v1/rounds/advance",
                       headers={"X-Admin-Key": "nope"}).status_code == 401
    assert client.post("/v1/rounds/advance").status_code == 401


def test_admin_check_uses_compare_digest(monkeypatch):
    """The admin check must go through hmac.compare_digest (constant time)."""
    import hmac

    from controlplane.server import app as app_module

    calls = {"n": 0}
    real = hmac.compare_digest

    def spy(a, b):
        calls["n"] += 1
        return real(a, b)

    monkeypatch.setattr(app_module.hmac, "compare_digest", spy)
    try:
        app_module.require_admin("whatever")
    except Exception:
        pass
    assert calls["n"] >= 1


# --------------------------------------------------------------------------- #
# Env-driven CORS allowlist (no blanket wildcard).
# --------------------------------------------------------------------------- #
def test_cors_is_not_wildcard():
    from controlplane.server import app as app_module
    assert "*" not in app_module._CORS_ORIGINS
    assert any("localhost" in o for o in app_module._CORS_ORIGINS)


# --------------------------------------------------------------------------- #
# /predict realness: indicators derive from real per-feature contributions.
# --------------------------------------------------------------------------- #
def test_predict_shape_and_real_indicators(plane):
    """A trained model yields indicators tied to the features actually driving
    the score, and the response carries the {score, indicators, regime} fields
    plus the legacy label/confidence."""
    _enroll(plane, ["n0", "n1", "n2"])
    plane.min_updates = 3
    # Move the campaignSig coordinate up via an honest round so that feature
    # gains a positive weight and a campaign txn lights it up as an indicator.
    base = np.zeros(DIM)
    base[9] = 0.5   # campaignSig coordinate
    for mid in ["n0", "n1", "n2"]:
        plane.submit_update(mid, 1, list(base), 1000, {"recall": 0.9})
    res = plane.advance_round()
    plane.promote(res.new_version)

    out = plane.demo_predict({"campaignSignature": 1.0})
    assert set(out) >= {"label", "confidence", "score", "indicators", "regime"}
    assert out["label"] in {"fraud", "legitimate"}
    assert 0.0 <= out["score"] <= 1.0
    assert out["regime"] == "federated"
    assert isinstance(out["indicators"], list) and out["indicators"]
    # campaignSig is now a positive-weight feature and the txn sets it high, so it
    # MUST surface as the (or an) indicator — not a hardcoded list.
    assert "matches active campaign signature" in out["indicators"]


def test_predict_indicators_vary_with_transaction(plane):
    """Different transactions can produce different indicator lists (proving the
    indicators are derived, not a fixed constant)."""
    _enroll(plane, ["n0", "n1", "n2"])
    plane.min_updates = 3
    base = np.zeros(DIM)
    base[0] = 0.5   # amount coordinate gets positive weight
    base[6] = 0.5   # velocity coordinate gets positive weight
    for mid in ["n0", "n1", "n2"]:
        plane.submit_update(mid, 1, list(base), 1000, {"recall": 0.9})
    res = plane.advance_round()
    plane.promote(res.new_version)

    only_amount = plane.demo_predict({"amount": 3.0})["indicators"]
    only_velocity = plane.demo_predict({"velocity": 3.0})["indicators"]
    assert "anomalous transfer amount" in only_amount
    assert "high transaction velocity" in only_velocity
    assert only_amount != only_velocity
