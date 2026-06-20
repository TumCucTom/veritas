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

    def attest(self) -> Quote:
        document = {
            "memberId": self.identity.member_id,
            "tenantId": self.identity.tenant_id,
            "imageId": self.image_id,
            "configDigest": self.config_digest,
        }
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        measurement = hashlib.sha256(canonical).hexdigest()
        sig = self.identity.sign(measurement.encode())
        return Quote(
            kind="software-stub",
            measurement=measurement,
            signature=base64.b64encode(sig).decode(),
            public_key_pem=self.identity.public_key_pem(),
            issued_at=time.time(),
            document=document,
        )
