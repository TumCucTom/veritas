"""SQLite-backed durable store for the control plane (stdlib ``sqlite3``).

The working state stays in memory for speed (see ``state.ControlPlane``); this
store is a **write-through** persistence layer so that members, models,
rounds/results, the transparency log, and the DP-spent counter survive a
process restart.

Design:
  * One SQLite file (env ``VERITAS_DB``, default ``controlplane.db``); tests use
    ``:memory:``, which is a no-op store that never persists across connections
    (so the in-memory tests behave exactly as before).
  * ``write-through``: every state mutation calls a ``persist_*`` method.
  * ``load_all()`` returns everything needed to rehydrate a ``ControlPlane`` on
    start, so a fresh process reconstructs the prior state from the same DB.

All JSON-able blobs (weight vectors, metrics, contributor lists, transparency
data) are stored as canonical JSON text.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading

DEFAULT_DB = "controlplane.db"


def _dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


class Store:
    """Write-through SQLite persistence for the control plane.

    A ``:memory:`` path yields an ephemeral DB scoped to this object — fine for
    tests, where each ``ControlPlane`` gets its own. A file path persists.
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or os.environ.get("VERITAS_DB", DEFAULT_DB)
        # check_same_thread=False because the plane is guarded by its own RLock
        # and FastAPI may touch it from worker threads.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    @property
    def is_memory(self) -> bool:
        return self.db_path == ":memory:"

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS members (
                    member_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    public_key_pem TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attestation_quote TEXT,
                    last_sync TEXT,
                    customers INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS models (
                    version INTEGER PRIMARY KEY,
                    weights TEXT NOT NULL,
                    parent_version INTEGER,
                    status TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rounds (
                    round INTEGER PRIMARY KEY,
                    new_version INTEGER NOT NULL,
                    contributors TEXT NOT NULL,
                    rejected TEXT NOT NULL,
                    global_metrics TEXT NOT NULL,
                    transparency_seq INTEGER NOT NULL,
                    silo_recall REAL
                );
                CREATE TABLE IF NOT EXISTS transparency (
                    seq INTEGER PRIMARY KEY,
                    type TEXT NOT NULL,
                    round INTEGER,
                    data TEXT NOT NULL,
                    leaf_hash TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            # Migration: add the customers column to pre-existing member tables.
            cols = {r["name"] for r in
                    self._conn.execute("PRAGMA table_info(members)")}
            if "customers" not in cols:
                self._conn.execute(
                    "ALTER TABLE members ADD COLUMN customers INTEGER NOT NULL "
                    "DEFAULT 0")
            self._conn.commit()

    # -- write-through persisters ----------------------------------------
    def persist_member(self, m) -> None:
        aq = m.attestation_quote
        aq_text = aq if isinstance(aq, str) or aq is None else _dumps(aq)
        with self._lock:
            self._conn.execute(
                """INSERT INTO members
                   (member_id, display_name, tenant_id, public_key_pem,
                    status, attestation_quote, last_sync, customers)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(member_id) DO UPDATE SET
                     display_name=excluded.display_name,
                     tenant_id=excluded.tenant_id,
                     public_key_pem=excluded.public_key_pem,
                     status=excluded.status,
                     attestation_quote=excluded.attestation_quote,
                     last_sync=excluded.last_sync,
                     customers=excluded.customers""",
                (m.member_id, m.display_name, m.tenant_id, m.public_key_pem,
                 m.status, aq_text, m.last_sync, int(m.customers)),
            )
            self._conn.commit()

    def persist_model(self, m) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO models
                   (version, weights, parent_version, status, metrics, created_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(version) DO UPDATE SET
                     weights=excluded.weights,
                     parent_version=excluded.parent_version,
                     status=excluded.status,
                     metrics=excluded.metrics,
                     created_at=excluded.created_at""",
                (m.version, _dumps(m.weights), m.parent_version, m.status,
                 _dumps(m.metrics), m.created_at),
            )
            self._conn.commit()

    def persist_round_result(self, res, silo_recall=None) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO rounds
                   (round, new_version, contributors, rejected, global_metrics,
                    transparency_seq, silo_recall)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(round) DO UPDATE SET
                     new_version=excluded.new_version,
                     contributors=excluded.contributors,
                     rejected=excluded.rejected,
                     global_metrics=excluded.global_metrics,
                     transparency_seq=excluded.transparency_seq,
                     silo_recall=excluded.silo_recall""",
                (res.round, res.new_version, _dumps(res.contributors),
                 _dumps(res.rejected), _dumps(res.global_metrics),
                 res.transparency_seq, silo_recall),
            )
            self._conn.commit()

    def persist_transparency(self, entry: dict) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO transparency
                   (seq, type, round, data, leaf_hash, timestamp)
                   VALUES (?,?,?,?,?,?)""",
                (entry["seq"], entry["type"], entry.get("round"),
                 _dumps(entry["data"]), entry["leafHash"], entry["timestamp"]),
            )
            self._conn.commit()

    def persist_meta(self, key: str, value) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                (key, _dumps(value)),
            )
            self._conn.commit()

    # -- load-on-start ----------------------------------------------------
    def load_all(self) -> dict:
        with self._lock:
            members = [dict(r) for r in self._conn.execute("SELECT * FROM members")]
            models = [dict(r) for r in
                      self._conn.execute("SELECT * FROM models ORDER BY version")]
            rounds = [dict(r) for r in
                      self._conn.execute("SELECT * FROM rounds ORDER BY round")]
            transparency = [dict(r) for r in
                            self._conn.execute(
                                "SELECT * FROM transparency ORDER BY seq")]
            meta = {r["key"]: json.loads(r["value"])
                    for r in self._conn.execute("SELECT * FROM meta")}
        # Decode JSON blobs.
        for m in models:
            m["weights"] = json.loads(m["weights"])
            m["metrics"] = json.loads(m["metrics"])
        for r in rounds:
            r["contributors"] = json.loads(r["contributors"])
            r["rejected"] = json.loads(r["rejected"])
            r["global_metrics"] = json.loads(r["global_metrics"])
        for t in transparency:
            t["data"] = json.loads(t["data"])
        return {
            "members": members,
            "models": models,
            "rounds": rounds,
            "transparency": transparency,
            "meta": meta,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
