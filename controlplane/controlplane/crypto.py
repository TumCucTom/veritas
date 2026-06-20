"""Cryptographic primitives for the control plane.

Ed25519 keypairs for the control plane and members, EdDSA-signed JWTs for
node->plane auth, canonical-JSON serialisation, and SHA-256 hashing for the
Merkle transparency log. We reuse the `cryptography` and `PyJWT` libraries
rather than rolling our own crypto.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os as _os
import uuid
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

# Audience/issuer the node->plane JWTs are minted for and verified against.
# Binding `aud`/`iss` stops a member token issued for one service being replayed
# against another. Overridable via env so a deployment can scope them per-fleet.
JWT_AUDIENCE = _os.environ.get("VERITAS_JWT_AUD", "veritas-control-plane")
JWT_ISSUER = _os.environ.get("VERITAS_JWT_ISS", "veritas-node")
# Shorter default TTL: a node->plane token only needs to live long enough to
# submit a round's worth of updates. Tightening the window limits replay value.
JWT_DEFAULT_TTL_SECONDS = int(_os.environ.get("VERITAS_JWT_TTL", "300"))


# ---------------------------------------------------------------------------
# Canonical JSON + hashing
# ---------------------------------------------------------------------------
def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON: sorted keys, no extra whitespace, UTF-8 bytes.

    Two structurally-equal records always serialise to identical bytes, so a
    leaf hash is stable across processes — the basis of the transparency log.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def leaf_hash(record: Any) -> str:
    """SHA-256 over canonical-JSON of a record, domain-separated (0x00 prefix)."""
    return sha256_hex(b"\x00" + canonical_json(record))


def node_hash(left: str, right: str) -> str:
    """Internal Merkle node hash, domain-separated (0x01 prefix)."""
    return sha256_hex(b"\x01" + bytes.fromhex(left) + bytes.fromhex(right))


# ---------------------------------------------------------------------------
# Ed25519 keypair helpers
# ---------------------------------------------------------------------------
def generate_keypair() -> tuple[str, str]:
    """Return (private_pem, public_pem) for a fresh Ed25519 keypair."""
    priv = Ed25519PrivateKey.generate()
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return priv_pem, pub_pem


def load_or_create_keypair(path: str) -> tuple[str, str]:
    """Return (private_pem, public_pem), loading from `path` or generating once.

    Persists the control-plane Ed25519 signing identity so the transparency log
    stays verifiable across restarts (an ephemeral key would invalidate every
    previously-published signed root). The PEM file is written ``0600``. A
    ``:memory:`` path is treated as ephemeral (generate, never persist) for
    tests.
    """
    import os
    if path == ":memory:":
        return generate_keypair()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            priv_pem = fh.read()
        return priv_pem, public_pem_of(priv_pem)
    priv_pem, pub_pem = generate_keypair()
    # Write 0600 (owner read/write only).
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(priv_pem)
    return priv_pem, pub_pem


def load_private_key(pem: str) -> Ed25519PrivateKey:
    return serialization.load_pem_private_key(pem.encode(), password=None)


def load_public_key(pem: str) -> Ed25519PublicKey:
    return serialization.load_pem_public_key(pem.encode())


def public_pem_of(priv_pem: str) -> str:
    return load_private_key(priv_pem).public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()


# ---------------------------------------------------------------------------
# Detached signatures (used to sign the Merkle root)
# ---------------------------------------------------------------------------
def sign_bytes(priv_pem: str, data: bytes) -> str:
    """Hex-encoded Ed25519 signature over `data`."""
    return load_private_key(priv_pem).sign(data).hex()


def verify_bytes(pub_pem: str, data: bytes, sig_hex: str) -> bool:
    try:
        load_public_key(pub_pem).verify(bytes.fromhex(sig_hex), data)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# EdDSA JWTs (member node -> plane auth)
# ---------------------------------------------------------------------------
def make_member_jwt(
    priv_pem: str, member_id: str, tenant_id: str,
    ttl_seconds: int = JWT_DEFAULT_TTL_SECONDS,
    audience: str = JWT_AUDIENCE, issuer: str = JWT_ISSUER,
) -> str:
    """Sign a node->plane JWT with the member's private key (EdDSA).

    The token carries `aud`/`iss` (so it can only be redeemed against this
    control plane) and a unique `jti` (so a single token is individually
    identifiable / revocable). TTL is short by default.
    """
    now = _dt.datetime.now(_dt.timezone.utc)
    claims = {
        "sub": member_id,
        "tid": tenant_id,
        "aud": audience,
        "iss": issuer,
        "jti": uuid.uuid4().hex,
        "iat": int(now.timestamp()),
        "exp": int((now + _dt.timedelta(seconds=ttl_seconds)).timestamp()),
    }
    return jwt.encode(claims, load_private_key(priv_pem), algorithm="EdDSA")


def verify_member_jwt(
    token: str, pub_pem: str,
    audience: str = JWT_AUDIENCE, issuer: str = JWT_ISSUER,
) -> dict:
    """Verify a node->plane JWT against the member's registered public key.

    Verifies signature, expiry, and the `aud`/`iss` binding, and requires the
    `sub`/`tid`/`exp`/`aud`/`iss`/`jti` claims to be present. Raises
    jwt.PyJWTError on any failure (bad signature, expired, wrong audience, etc.).
    """
    return jwt.decode(
        token,
        load_public_key(pub_pem),
        algorithms=["EdDSA"],
        audience=audience,
        issuer=issuer,
        options={"require": ["sub", "tid", "exp", "aud", "iss", "jti"]},
    )
