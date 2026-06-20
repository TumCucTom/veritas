"""Node runtime: composes identity, engine, connector and federation client,
and runs the background federation loop. The FastAPI app holds one of these.

The federation loop is a plain background thread so the node keeps serving
predictions (local inference is always available, even if the plane is down —
the graceful-degradation requirement). Each round publishes SSE events.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from .attestation import SoftwareAttestor
from .config import NodeConfig
from .connectors import load_feature_map
from .connectors.runtime import ConnectorRuntime
from .engine import NodeEngine
from .federation import FederationClient, HttpxTransport
from .federation.transport import PlaneTransport
from .identity import NodeIdentity


class NodeRuntime:
    def __init__(
        self,
        cfg: NodeConfig,
        *,
        transport: PlaneTransport | None = None,
        event_sink: Callable[[dict], None] | None = None,
    ):
        self.cfg = cfg
        self.identity = NodeIdentity.generate(cfg.node_id, cfg.tenant_id)
        self.attestor = SoftwareAttestor(
            self.identity,
            image_id=f"veritas-node:{cfg.node_id}",
            config_digest=cfg.feature_map_path,
        )

        # Connector data is optional — fall back to synthetic if the map/source
        # is missing, so the node always boots.
        connector_data = self._load_connector_data()
        self.engine = NodeEngine(cfg, connector_data=connector_data)

        self.transport: PlaneTransport = transport or HttpxTransport(cfg.plane_url)
        self.client = FederationClient(
            self.identity, self.transport, self.attestor,
            rng=np.random.default_rng(cfg.seed + 7),
            # Enrol with REAL bank metadata so the plane's banks[] shows the
            # live institution (name + customer count) the engine represents.
            display_name=self.engine.name,
            customers=self.engine.customers,
            # Demo-control reactions: the client reads campaignActive/epoch off
            # the poll response and calls these to mutate LOCAL engine state.
            on_campaign=self._on_campaign,
            on_reset=self._on_reset,
        )

        self._event_sink = event_sink or (lambda ev: None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.plane_connected = False
        self.last_error: str | None = None

    def _load_connector_data(self):
        try:
            fmap = load_feature_map(self.cfg.feature_map_path)
            X, y = ConnectorRuntime(fmap).load()
            return (X, y)
        except Exception as exc:  # missing/invalid map → synthetic fallback
            self.last_error = f"connector fallback: {exc}"
            return None

    # ---- demo-control reactions (wired into the federation poll loop) -----

    def _is_blind(self) -> bool:
        """Whether THIS node is the designated blind node for the demo.

        A blind node injects the campaign EVAL-ONLY so its siloed baseline stays
        blind to the typology — only the FEDERATED model (learning from seeing
        peers) detects it. That gap is the measurable federated-vs-siloed lift.
        """
        return self.cfg.blind_node is not None and self.cfg.blind_node == self.cfg.node_index

    def _on_campaign(self, epoch: int) -> None:
        """Inject the cross-institution campaign locally (once per epoch).

        Preserves the seeing/blind behaviour: a blind node injects eval-only
        (``seeing=False``); every other node injects ``seeing=True`` so its
        siloed model can learn the typology from local examples.
        """
        seeing = not self._is_blind()
        self.engine.inject_campaign(seeing=seeing)
        self._emit("campaign_injected", {
            "bankId": self.cfg.node_id, "epoch": epoch, "seeing": seeing,
        })

    def _on_reset(self, epoch: int) -> None:
        """Reset local engine state to genesis (epoch bump → demo re-run)."""
        self.engine.reset_genesis()
        self._emit("epoch_reset", {"bankId": self.cfg.node_id, "epoch": epoch})

    # ---- event helpers ---------------------------------------------------

    def set_event_sink(self, sink: Callable[[dict], None]) -> None:
        self._event_sink = sink

    def _emit(self, type_: str, data: dict) -> None:
        self._event_sink({"type": type_, "data": data})

    # ---- enrolment (register identity so an admin can approve) -----------

    def ensure_enrolled(self) -> dict | None:
        """Register this node's identity with the plane (idempotent, best-effort).

        Called on startup so the member appears as ``pending`` immediately and an
        admin can approve it BEFORE federation begins — independent of whether
        the federation loop autostarts. Safe if the plane is down (the loop will
        enrol lazily on its first round); a 409 (already enrolled) is benign.
        """
        if self.client.enrolled:
            return None
        try:
            resp = self.client.enroll()
            self.plane_connected = True
            self.last_error = None
            return resp
        except Exception as exc:  # plane down or already enrolled → tolerate
            import httpx
            if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None \
                    and exc.response.status_code == 409:
                # Already enrolled in a prior run of this process's plane.
                self.client.enrolled = True
                return None
            self.last_error = f"enroll deferred: {exc}"
            return None

    # ---- one federation round (also callable directly in tests) ----------

    def federate_once(self) -> dict:
        data, y = self.engine.train_data, self.engine.train_y
        # Advance the siloed counterfactual FIRST so siloRecall is a freshly
        # measured number off the trained baseline (honest-metrics contract).
        self.engine.train_silo_step()
        silo_recall = self.engine.silo_recall()
        result = self.client.run_round(data, y, silo_recall=silo_recall)
        # Sync the new global model back into the engine.
        gw, ver = self.client.pull_global()
        self.engine.set_global(gw, ver)
        self.engine.accrue_counters()
        self.plane_connected = True
        self.last_error = None

        self._emit("client_updated", {
            "bankId": self.cfg.node_id,
            "detection": self.engine.state()["banks"][0]["detection"],
        })
        self._emit("round_complete", self.engine.state())
        return {
            "round": result.round,
            "submitted": result.submitted,
            "localRecall": result.local_recall,
            "globalVersion": result.global_version_after,
            "updateNorm": result.update_norm,
        }

    # ---- background loop -------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.federate_once()
            except Exception as exc:  # plane down → keep serving locally
                self.plane_connected = False
                self.last_error = str(exc)
            self._stop.wait(self.cfg.poll_interval)

    def start_federation(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="veritas-federation", daemon=True)
        self._thread.start()

    def stop_federation(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def health(self) -> dict:
        quote = self.attestor.attest()
        return {
            "ok": True,
            "memberId": self.cfg.node_id,
            "tenantId": self.cfg.tenant_id,
            "modelVersion": self.engine.global_version,
            "controlPlane": "connected" if self.plane_connected else "down",
            "attestation": {"kind": quote.kind, "measurement": quote.measurement[:16]},
            "lastError": self.last_error,
        }
