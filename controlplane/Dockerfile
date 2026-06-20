# Veritas Tier-2 Federation Control Plane (:9000)
#
# BUILD CONTEXT MUST BE THE REPO ROOT (the image needs ./core too):
#   docker build -f controlplane/Dockerfile -t veritas-controlplane .
# docker-compose.yml sets `context: ..` so this resolves automatically.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Reused FL primitives (core) + the control plane, both installed editable so
# `from veritas_core...` resolves exactly as it does locally.
COPY core /app/core
COPY controlplane /app/controlplane

# Drop any host virtualenvs/caches that rode along in the build context so the
# image stays lean and uses its own interpreter.
RUN rm -rf /app/core/.venv /app/controlplane/.venv \
           /app/core/**/__pycache__ /app/controlplane/**/__pycache__ \
 && pip install --upgrade pip \
 && pip install -e /app/core \
 && pip install -e /app/controlplane

EXPOSE 9000

# VERITAS_ADMIN_KEY and VERITAS_MIN_UPDATES are supplied via compose/env.
CMD ["uvicorn", "controlplane.server.app:app", "--host", "0.0.0.0", "--port", "9000"]
