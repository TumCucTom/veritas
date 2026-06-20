"""Append-only, signed Merkle transparency log.

Each appended record is wrapped as a leaf (SHA-256 of canonical-JSON). The
Merkle root over all leaves is signed by the control-plane Ed25519 key. We
expose inclusion (audit) proofs so any party can verify a record is committed
under a signed root without trusting the server.

The Merkle tree uses RFC 6962 / Certificate-Transparency style hashing:
duplicate-promotion of an odd trailing node at each level. `node_hash` and
`leaf_hash` are domain-separated in `crypto.py`.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any

from .crypto import canonical_json, leaf_hash, node_hash, sign_bytes


class TransparencyLog:
    def __init__(self, plane_priv_pem: str):
        self._priv_pem = plane_priv_pem
        self._records: list[dict] = []  # each: seq, type, round?, data, leafHash, timestamp
        self._leaves: list[str] = []    # leaf hashes, parallel to _records

    # -- append -----------------------------------------------------------
    def append(self, rec_type: str, data: dict, round_: int | None = None) -> dict:
        seq = len(self._records)
        ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
        # The signed leaf commits to the record content (seq/type/round/data/ts).
        canonical_record: dict[str, Any] = {
            "seq": seq,
            "type": rec_type,
            "data": data,
            "timestamp": ts,
        }
        if round_ is not None:
            canonical_record["round"] = round_
        lh = leaf_hash(canonical_record)
        entry = dict(canonical_record)
        entry["leafHash"] = lh
        self._records.append(entry)
        self._leaves.append(lh)
        return entry

    # -- reads ------------------------------------------------------------
    def list(self) -> list[dict]:
        return list(self._records)

    def size(self) -> int:
        return len(self._records)

    # -- Merkle tree ------------------------------------------------------
    def _levels(self) -> list[list[str]]:
        """Return all levels bottom-up; levels[0] == leaves, last == [root]."""
        if not self._leaves:
            return [[]]
        levels = [list(self._leaves)]
        cur = levels[0]
        while len(cur) > 1:
            nxt = []
            for i in range(0, len(cur), 2):
                left = cur[i]
                right = cur[i + 1] if i + 1 < len(cur) else cur[i]  # promote odd tail
                nxt.append(node_hash(left, right))
            levels.append(nxt)
            cur = nxt
        return levels

    def root_hash(self) -> str | None:
        if not self._leaves:
            return None
        return self._levels()[-1][0]

    def signed_root(self) -> dict:
        rh = self.root_hash()
        size = len(self._leaves)
        if rh is None:
            return {"size": 0, "rootHash": None, "signaturePem": None}
        # Sign canonical-JSON of {size, rootHash} so the signature is bound to
        # the tree size as well as the root (prevents root/size mismatch).
        payload = canonical_json({"size": size, "rootHash": rh})
        sig = sign_bytes(self._priv_pem, payload)
        return {"size": size, "rootHash": rh, "signaturePem": sig}

    def inclusion_proof(self, seq: int) -> dict:
        """Audit path proving leaf `seq` is committed under the current root."""
        if seq < 0 or seq >= len(self._leaves):
            raise IndexError("seq out of range")
        levels = self._levels()
        path: list[str] = []
        idx = seq
        for level in levels[:-1]:  # exclude root level
            sibling = idx ^ 1
            if sibling < len(level):
                path.append(level[sibling])
            else:
                path.append(level[idx])  # promoted self (odd tail)
            idx //= 2
        return {
            "seq": seq,
            "leafHash": self._leaves[seq],
            "rootHash": self.root_hash(),
            "auditPath": path,
            "index": seq,
        }


def verify_inclusion_proof(proof: dict, size: int) -> bool:
    """Recompute the root from a leaf hash + audit path; compare to claimed root.

    `size` is the tree size at proof time (needed to mirror odd-tail promotion).
    """
    idx = proof["index"]
    h = proof["leafHash"]
    level_size = size
    for sibling in proof["auditPath"]:
        if idx % 2 == 0:
            # left child; right sibling is the recorded one, unless it was the
            # promoted self (odd tail), in which case sibling == h.
            h = node_hash(h, sibling)
        else:
            h = node_hash(sibling, h)
        idx //= 2
        level_size = (level_size + 1) // 2
    return h == proof["rootHash"]
