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
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


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
    priv_pem: str, member_id: str, tenant_id: str, ttl_seconds: int = 3600
) -> str:
    """Sign a node->plane JWT with the member's private key (EdDSA)."""
    now = _dt.datetime.now(_dt.timezone.utc)
    claims = {
        "sub": member_id,
        "tid": tenant_id,
        "iat": int(now.timestamp()),
        "exp": int((now + _dt.timedelta(seconds=ttl_seconds)).timestamp()),
    }
    return jwt.encode(claims, load_private_key(priv_pem), algorithm="EdDSA")


def verify_member_jwt(token: str, pub_pem: str) -> dict:
    """Verify a node->plane JWT against the member's registered public key.

    Raises jwt.PyJWTError on any failure (bad signature, expired, etc.).
    """
    return jwt.decode(
        token,
        load_public_key(pub_pem),
        algorithms=["EdDSA"],
        options={"require": ["sub", "tid", "exp"]},
    )
