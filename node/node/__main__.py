"""CLI runner: ``python -m node`` starts a node via uvicorn.

Flags mirror the env vars (CLI takes precedence). Example:
    python -m node --node-index 0 --port 8100 --plane-url http://localhost:9000
"""
from __future__ import annotations

import argparse
import os


def main() -> None:
    p = argparse.ArgumentParser(prog="node", description="Run a Veritas Tier 1 Bank Node")
    p.add_argument("--node-id")
    p.add_argument("--node-index", type=int)
    p.add_argument("--tenant-id")
    p.add_argument("--port", type=int)
    p.add_argument("--plane-url")
    p.add_argument("--feature-map")
    p.add_argument("--seed", type=int)
    p.add_argument("--no-federation", action="store_true", help="don't autostart the federation loop")
    args = p.parse_args()

    # CLI -> env so config.from_env() (used by the app module) picks them up.
    if args.node_index is not None:
        os.environ["VERITAS_NODE_INDEX"] = str(args.node_index)
    if args.node_id:
        os.environ["VERITAS_NODE_ID"] = args.node_id
    if args.tenant_id:
        os.environ["VERITAS_TENANT_ID"] = args.tenant_id
    if args.plane_url:
        os.environ["VERITAS_PLANE_URL"] = args.plane_url
    if args.feature_map:
        os.environ["VERITAS_FEATURE_MAP"] = args.feature_map
    if args.seed is not None:
        os.environ["VERITAS_SEED"] = str(args.seed)
    if args.no_federation:
        os.environ["VERITAS_AUTOSTART_FEDERATION"] = "0"

    idx = int(os.environ.get("VERITAS_NODE_INDEX", "0"))
    port = args.port if args.port is not None else int(os.environ.get("VERITAS_PORT", str(8100 + idx)))
    os.environ["VERITAS_PORT"] = str(port)

    import uvicorn

    uvicorn.run("node.server.app:app", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
