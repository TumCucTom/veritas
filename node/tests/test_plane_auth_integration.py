"""Integration: a node-minted JWT must satisfy the control plane's verifier.

This is the one seam isolated suites miss — node tests use a fake plane and the
control-plane suite mints its own tokens. After hardening verify_member_jwt to
require aud/iss/jti, node.identity.mint_jwt must emit them or every real
node->plane request 401s.
"""
import jwt
import pytest

from node.identity import NodeIdentity

crypto = pytest.importorskip("controlplane.crypto")


def test_node_jwt_passes_plane_verifier():
    ident = NodeIdentity.generate("bank0", "tenant0")
    token = ident.mint_jwt()
    claims = crypto.verify_member_jwt(token, ident.public_key_pem())
    assert claims["sub"] == "bank0"
    assert claims["tid"] == "tenant0"
    assert claims["jti"]


def test_node_jwt_rejected_for_wrong_audience():
    ident = NodeIdentity.generate("bank0", "tenant0")
    token = ident.mint_jwt()
    with pytest.raises(jwt.PyJWTError):
        crypto.verify_member_jwt(token, ident.public_key_pem(), audience="other-plane")
