# Veritas — Tier 1 Bank Node

One bank's deployable node: holds local data, trains locally, federates
DP-protected model deltas with the control plane over HTTP, serves predictions,
and secure-aggregates Tier-0 device updates. Splits the in-process N-bank
`core/veritas_core/engine.py` into ONE bank + a networked federation client.
All FL math (model/dp/aggregation) is reused from `core/veritas_core`.

## Setup
```bash
cd node
python3.13 -m venv .venv && . .venv/bin/activate
pip install -e .          # installs the node
pip install -e ../core    # makes `from veritas_core...` imports work
pip install -e ".[dev]"   # pytest
python -m pytest -q
```

## Run a node
Node `i` listens on port `8100 + i`.
```bash
# env-var form (matches the prompt):
VERITAS_NODE_ID=node0 VERITAS_NODE_INDEX=0 VERITAS_PORT=8100 \
  VERITAS_PLANE_URL=http://localhost:9000 \
  uvicorn node.server.app:app --port 8100

# or the CLI runner (CLI flags override env):
python -m node --node-index 0 --port 8100 --plane-url http://localhost:9000
```

### Env vars / CLI flags
| env | flag | default |
|---|---|---|
| `VERITAS_NODE_INDEX` | `--node-index` | 0 |
| `VERITAS_NODE_ID` | `--node-id` | `node{index}` |
| `VERITAS_TENANT_ID` | `--tenant-id` | `tenant{index}` |
| `VERITAS_PORT` | `--port` | `8100 + index` |
| `VERITAS_PLANE_URL` | `--plane-url` | `http://localhost:9000` |
| `VERITAS_FEATURE_MAP` | `--feature-map` | `sample_data/feature_map.yaml` |
| `VERITAS_SEED` | `--seed` | `index` |
| `VERITAS_AUTOSTART_FEDERATION` | `--no-federation` | on |

The federation loop runs in a background thread; if the control plane is
unreachable the node keeps serving predictions on the last-good model
(`/health` reports `controlPlane: down`).

## Endpoints
Outward (bank systems): `POST /predict`, `GET /state`, `GET /events` (SSE),
`GET /health`. Edge (Tier 0): `GET /edge/v1/model`, `POST /edge/v1/updates`,
`POST /edge/v1/score`. Dev: `POST /federate/step`, `POST /campaign/inject`.

## Connector / feature map (config, not code)
`sample_data/feature_map.yaml` maps a source onto the FEATURE_DIM contract.
Source kinds: `csv` (warehouse stand-in) and `iso20022` (pacs.008/pain.001 XML,
see `feature_map_iso20022.yaml`). The runtime emits `(X, y)` for the trainer.
