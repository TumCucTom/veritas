# Veritas Tier-1 Bank Node (:8100+i)
#
# BUILD CONTEXT MUST BE THE REPO ROOT (the image needs ./core too):
#   docker build -f node/Dockerfile -t veritas-node .
# docker-compose.yml sets `context: ..` so this resolves automatically.
#
# One image serves any node; identity is parameterised at runtime via
# VERITAS_NODE_ID / VERITAS_NODE_INDEX / VERITAS_PORT / VERITAS_PLANE_URL.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Reused FL primitives (core) + the node package. `sample_data/` (the connector
# feature map + transactions.csv) ships with the node source tree and is found
# relative to the package via __file__, so an editable install keeps it intact.
COPY core /app/core
COPY node /app/node

# Drop any host virtualenvs/caches that rode along in the build context so the
# image stays lean and uses its own interpreter.
RUN rm -rf /app/core/.venv /app/node/.venv \
           /app/core/**/__pycache__ /app/node/**/__pycache__ \
 && pip install --upgrade pip \
 && pip install -e /app/core \
 && pip install -e /app/node

# Default node identity; override per-replica in compose.
ENV VERITAS_NODE_ID=node0 \
    VERITAS_NODE_INDEX=0 \
    VERITAS_PORT=8100 \
    VERITAS_PLANE_URL=http://controlplane:9000 \
    VERITAS_AUTOSTART_FEDERATION=1 \
    VERITAS_POLL_INTERVAL=3

EXPOSE 8100-8105

# The port is taken from VERITAS_PORT (set per replica) at container start.
CMD ["sh", "-c", "uvicorn node.server.app:app --host 0.0.0.0 --port ${VERITAS_PORT}"]
