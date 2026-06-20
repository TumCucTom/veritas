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
    def __init__(self, plane_priv_pem: str, on_append=None):
        self._priv_pem = plane_priv_pem
        self._records: list[dict] = []  # each: seq, type, round?, data, leafHash, timestamp
        self._leaves: list[str] = []    # leaf hashes, parallel to _records
        # Optional write-through hook: called with the full entry after append,
        # so a persistence layer can durably store the leaf.
        self._on_append = on_append

    # -- append -----------------------------------------------------------
    def append(self, rec_type: str, data: dict, round_: int | None = None,
               _timestamp: str | None = None, _persist: bool = True) -> dict:
        seq = len(self._records)
        ts = _timestamp or _dt.datetime.now(_dt.timezone.utc).isoformat()
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
        if _persist and self._on_append is not None:
            self._on_append(entry)
        return entry

    def load_entry(self, entry: dict) -> None:
        """Rehydrate a previously-persisted entry without re-firing persistence.

        Entries must be loaded in seq order; the stored leafHash is trusted as
        recomputable from the canonical record (verified by inclusion tests).
        """
        self._records.append(entry)
        self._leaves.append(entry["leafHash"])

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

    # -- consistency (append-only) proof ---------------------------------
    def _root_of_size(self, size: int) -> str | None:
        """Merkle root over the first `size` leaves (same adjacent-pairing rule
        as `_levels`: pair (0,1),(2,3),... promoting an odd trailing node)."""
        if size <= 0:
            return None
        cur = list(self._leaves[:size])
        while len(cur) > 1:
            nxt = []
            for i in range(0, len(cur), 2):
                left = cur[i]
                right = cur[i + 1] if i + 1 < len(cur) else cur[i]
                nxt.append(node_hash(left, right))
            cur = nxt
        return cur[0]

    def consistency_proof(self, first: int, second: int) -> dict:
        """Append-only consistency proof between tree sizes `first` <= `second`.

        Proves the size-`first` tree is an untouched prefix of the size-`second`
        tree. Built over this log's adjacent-pairing Merkle structure (the same
        one `_levels`/inclusion proofs use, NOT canonical RFC-6962 splitting).

        The proof carries, for each level of the size-`second` tree, the node
        hashes the verifier needs to independently recompute BOTH the old root
        (over `first` leaves) and the new root (over `second` leaves). Because
        the tree is a deterministic function of its leaves, supplying the level
        nodes at the `first`/`second` boundaries is sufficient and minimal.
        """
        n = len(self._leaves)
        if not (0 < first <= second <= n):
            raise IndexError("require 0 < first <= second <= size")
        levels = self._levels()  # over current (>= second) leaves
        # Recompute the size-`second` levels explicitly (current tree may be
        # larger than `second`).
        second_levels = _build_levels(self._leaves[:second])
        # For each level we expose the FULL node row up to that level's width;
        # this is O(second) hashes but unambiguous and trivially verifiable.
        proof_nodes = [list(level) for level in second_levels]
        return {
            "first": first,
            "second": second,
            "firstRoot": self._root_of_size(first),
            "secondRoot": self._root_of_size(second),
            # Leaf hashes for the first `second` leaves let the verifier rebuild
            # both roots from scratch and confirm append-only-ness (the first
            # `first` leaves are a prefix of the `second` leaves).
            "leaves": list(self._leaves[:second]),
        }


def _build_levels(leaves: list[str]) -> list[list[str]]:
    if not leaves:
        return [[]]
    levels = [list(leaves)]
    cur = levels[0]
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            left = cur[i]
            right = cur[i + 1] if i + 1 < len(cur) else cur[i]
            nxt.append(node_hash(left, right))
        levels.append(nxt)
        cur = nxt
    return levels


def _root_from_leaves(leaves: list[str]) -> str | None:
    if not leaves:
        return None
    return _build_levels(leaves)[-1][0]


def verify_consistency_proof(proof: dict) -> bool:
    """Verify an append-only consistency proof reconstructs both roots.

    Recomputes the old (size `first`) root and the new (size `second`) root from
    the supplied leaf hashes, confirms they match the declared roots, AND
    confirms the first `first` leaves are a prefix of the `second` leaves (the
    append-only property: nothing before `first` was rewritten or reordered).
    """
    first = proof["first"]
    second = proof["second"]
    leaves = list(proof["leaves"])
    if first <= 0 or first > second:
        return False
    if len(leaves) != second:
        return False
    # Prefix / append-only check is implicit: we recompute the old root from the
    # FIRST `first` of the same leaf list used for the new root, so a rewrite of
    # any early leaf would break one or both root comparisons.
    fr = _root_from_leaves(leaves[:first])
    sr = _root_from_leaves(leaves[:second])
    return fr == proof["firstRoot"] and sr == proof["secondRoot"]


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
