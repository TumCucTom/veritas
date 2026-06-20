"""Node identity: Ed25519 keypair + EdDSA-signed JWT minting.

Per PROTOCOL.md "Auth & identity": each member (bank node) has an Ed25519
keypair. Enrolment registers the public key under a tenant. Every authenticated
node→plane request carries ``Authorization: Bearer <jwt>`` where the JWT is
signed by the member's **private** key (EdDSA) with claims
``{sub: memberId, tid: tenantId, iat, exp}``. The control plane verifies the
signature against the registered public key.

The node mints its own tokens locally (it holds the private key); the plane is
the verifier. This keeps enrolment a one-time public-key registration.
"""
from __future__ import annotations

import time

import jwt  # PyJWT
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


class NodeIdentity:
    def __init__(self, member_id: str, tenant_id: str, private_key: Ed25519PrivateKey):
        self.member_id = member_id
        self.tenant_id = tenant_id
        self._sk = private_key
        self._pk: Ed25519PublicKey = private_key.public_key()

    # ---- construction ----------------------------------------------------

    @classmethod
    def generate(cls, member_id: str, tenant_id: str) -> "NodeIdentity":
        return cls(member_id, tenant_id, Ed25519PrivateKey.generate())

    @classmethod
    def from_private_pem(cls, member_id: str, tenant_id: str, pem: bytes) -> "NodeIdentity":
        sk = serialization.load_pem_private_key(pem, password=None)
        if not isinstance(sk, Ed25519PrivateKey):
            raise TypeError("expected an Ed25519 private key")
        return cls(member_id, tenant_id, sk)

    # ---- key material ----------------------------------------------------

    def public_key_pem(self) -> str:
        return self._pk.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()

    def private_key_pem(self) -> bytes:
        return self._sk.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def sign(self, data: bytes) -> bytes:
        return self._sk.sign(data)

    # ---- JWT -------------------------------------------------------------

    def mint_jwt(self, ttl_seconds: int = 3600) -> str:
        """Mint an EdDSA-signed JWT the plane verifies against the registered key."""
        now = int(time.time())
        claims = {
            "sub": self.member_id,
            "tid": self.tenant_id,
            "iat": now,
            "exp": now + ttl_seconds,
        }
        # PyJWT serialises the Ed25519 private key object directly for EdDSA.
        return jwt.encode(claims, self._sk, algorithm="EdDSA")

    @staticmethod
    def verify_jwt(token: str, public_key_pem: str) -> dict:
        """Verify a node JWT against a registered public key (the plane's job;
        provided here so the in-memory fake plane and tests share one impl)."""
        pk = serialization.load_pem_public_key(public_key_pem.encode())
        return jwt.decode(token, pk, algorithms=["EdDSA"])
