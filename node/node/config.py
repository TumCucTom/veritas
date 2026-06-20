"""Node configuration — parameterised by env vars (or CLI for the runner).

A node is one bank. It is identified by ``VERITAS_NODE_ID`` (the memberId on
the wire), seeded deterministically by ``VERITAS_NODE_INDEX`` so local data is
reproducible/auditable, and it dials a control plane at ``VERITAS_PLANE_URL``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Wire dimension = FEATURE_DIM + 1 (logistic bias). Reused from core.
from veritas_core.data import FEATURE_DIM  # noqa: E402

DEFAULT_FEATURE_MAP = str(Path(__file__).resolve().parent.parent / "sample_data" / "feature_map.yaml")

# DP parameters mirror core/veritas_core/engine.py so node updates match the
# control plane's expected clipping/noise regime. The plane also advertises
# dpParams in /v1/rounds/current; the node prefers those when present.
MAX_NORM = 2.0
SIGMA = 0.05
EPOCHS = 8
LR = 0.3


@dataclass
class NodeConfig:
    node_id: str = "node0"
    node_index: int = 0
    tenant_id: str = "tenant0"
    display_name: str = "Bank Node 0"
    port: int = 8100
    plane_url: str = "http://localhost:9000"
    feature_map_path: str = DEFAULT_FEATURE_MAP
    seed: int = 0
    # Polling cadence (seconds) for the background federation loop.
    poll_interval: float = 5.0
    # Start the federation loop automatically on app startup.
    autostart_federation: bool = True
    admin_key: str = ""
    # If set to this node's index, the node injects the campaign EVAL-ONLY
    # (``inject_campaign(seeing=False)``): its siloed baseline stays blind to the
    # new typology so only the FEDERATED model (learning from seeing peers)
    # detects it — the measurable federated-vs-siloed lift in the live demo.
    blind_node: int | None = None

    @property
    def dim(self) -> int:
        return FEATURE_DIM + 1

    @classmethod
    def from_env(cls) -> "NodeConfig":
        idx = int(os.environ.get("VERITAS_NODE_INDEX", "0"))
        node_id = os.environ.get("VERITAS_NODE_ID", f"node{idx}")
        return cls(
            node_id=node_id,
            node_index=idx,
            tenant_id=os.environ.get("VERITAS_TENANT_ID", f"tenant{idx}"),
            display_name=os.environ.get("VERITAS_DISPLAY_NAME", f"Bank Node {idx}"),
            port=int(os.environ.get("VERITAS_PORT", str(8100 + idx))),
            plane_url=os.environ.get("VERITAS_PLANE_URL", "http://localhost:9000"),
            feature_map_path=os.environ.get("VERITAS_FEATURE_MAP", DEFAULT_FEATURE_MAP),
            seed=int(os.environ.get("VERITAS_SEED", str(idx))),
            poll_interval=float(os.environ.get("VERITAS_POLL_INTERVAL", "5.0")),
            autostart_federation=os.environ.get("VERITAS_AUTOSTART_FEDERATION", "1") not in ("0", "false", "False"),
            admin_key=os.environ.get("VERITAS_ADMIN_KEY", ""),
            blind_node=(
                int(os.environ["VERITAS_BLIND_NODE"])
                if os.environ.get("VERITAS_BLIND_NODE", "").strip() != ""
                else None
            ),
        )
