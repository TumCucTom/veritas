"""Attestation agent.

Production runs the node inside a confidential-computing enclave (AWS Nitro
Enclaves / AMD SEV-SNP / Intel TDX) and ``attest()`` returns the hardware's
**remote attestation quote** — a signed measurement proving the exact sealed
image is running and cannot exfiltrate data. That quote is what converts
"trust us" into "verify us" for a bank's InfoSec.

This module ships a **software stub** for dev/test: it measures the running
image/config (SHA-256 over an identity document) and signs it with the node's
Ed25519 key. The interface is identical, so production swaps the implementation
without touching callers. The quote is attached to enrolment and to each round
update, and recorded in the control plane's transparency log.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .identity import NodeIdentity


@dataclass
class Quote:
    """An attestation quote. ``kind`` distinguishes the stub from a real TEE."""

    kind: str            # "software-stub" | "nitro" | "sev-snp" | "tdx"
    measurement: str     # hex SHA-256 over the identity document
    signature: str       # base64 Ed25519 signature over the measurement
    public_key_pem: str  # the signer's public key
    issued_at: float
    document: dict

    def to_wire(self) -> dict:
        return {
            "kind": self.kind,
            "measurement": self.measurement,
            "signature": self.signature,
            "publicKeyPem": self.public_key_pem,
            "issuedAt": self.issued_at,
            "document": self.document,
        }


class Attestor(Protocol):
    def attest(self) -> Quote: ...


class SoftwareAttestor:
    """Stub attestor: hash-of-image/config signed by the node key.

    PRODUCTION: replace with NitroAttestor / SevSnpAttestor / TdxAttestor that
    return the platform's signed quote over the same identity document plus the
    enclave's PCR/launch measurements. Callers and wire format are unchanged.
    """

    def __init__(self, identity: NodeIdentity, image_id: str, config_digest: str):
        self.identity = identity
        self.image_id = image_id
        self.config_digest = config_digest

    def _config_digest(self) -> str:
        """SHA-256 over the REAL config the node runs (the feature-map digest is
        passed in as ``config_digest``). Hashing it here means the measurement is
        bound to actual configuration, not an opaque caller-supplied string."""
        return hashlib.sha256(self.config_digest.encode()).hexdigest()

    def attest(self) -> Quote:
        # The measurement document carries REAL measurement fields (member /
        # tenant identity + image id + a digest of the running config), not a
        # bare string + a file path. The measurement is the SHA-256 over the
        # canonical document, and the signature is over that measurement, so the
        # plane can RE-DERIVE the measurement from the document and verify it was
        # signed by the enrolment key — a node cannot self-assert an arbitrary
        # measurement (see :func:`verify_quote`).
        document = {
            "memberId": self.identity.member_id,
            "tenantId": self.identity.tenant_id,
            "imageId": self.image_id,
            "configDigest": self._config_digest(),
        }
        measurement = _measure(document)
        sig = self.identity.sign(measurement.encode())
        return Quote(
            kind="software-stub",
            measurement=measurement,
            signature=base64.b64encode(sig).decode(),
            public_key_pem=self.identity.public_key_pem(),
            issued_at=time.time(),
            document=document,
        )


def _measure(document: dict) -> str:
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def verify_quote(quote: dict, *, enrolment_public_key_pem: str,
                 expected_member_id: str | None = None) -> None:
    """Verify an attestation quote the way the control plane must.

    RAISES on a forged/self-asserted/tampered quote; returns ``None`` on success.

    Checks performed (this is what turns "presence => verified" into real
    verification):
      1. The ``measurement`` is RE-DERIVED from ``document`` and must match the
         claimed measurement — a node cannot assert a measurement that does not
         hash from its own document.
      2. The Ed25519 ``signature`` over the measurement must verify against the
         key registered AT ENROLMENT (``enrolment_public_key_pem``), NOT the key
         embedded in the quote — otherwise a forger could ship its own key and
         "self-verify". Binding to the enrolment identity is the whole point.
      3. The document's ``memberId`` must match the enrolled member.

    LIMITATIONS (software stub): this proves the quote was signed by the
    enrolled member's key over a document binding its config digest — it does
    NOT prove a genuine TEE/hardware root of trust. A production
    NitroAttestor/SevSnpAttestor returns a quote whose signature chains to the
    platform's attestation CA (PCRs / launch measurement); this verifier's
    structure (re-derive measurement, verify signature over it against a trusted
    key, bind identity) is unchanged — only the trusted key becomes the platform
    CA instead of the enrolment key.
    """
    document = quote.get("document")
    if not isinstance(document, dict):
        raise ValueError("attestation quote missing document")
    claimed = quote.get("measurement")
    rederived = _measure(document)
    if claimed != rederived:
        raise ValueError("attestation measurement does not match its document (forged)")
    if expected_member_id is not None and document.get("memberId") != expected_member_id:
        raise ValueError("attestation document memberId does not match the enrolled member")
    try:
        sig = base64.b64decode(quote["signature"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("attestation signature is missing or malformed") from exc
    pk = serialization.load_pem_public_key(enrolment_public_key_pem.encode())
    if not isinstance(pk, Ed25519PublicKey):
        raise TypeError("enrolment key is not an Ed25519 public key")
    try:
        pk.verify(sig, rederived.encode())
    except InvalidSignature as exc:
        raise ValueError("attestation signature does not verify against the enrolment key") from exc
