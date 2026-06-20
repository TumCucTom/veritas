"""Federation client (node→plane) + injectable HTTP transport.

The client enrols once, then loops: pull current round + global model, train
locally on connector data, compute ``update = local_w - global_w``, DP-privatize
it, submit it, and pull the new global model back.

The HTTP transport is injected so the client is fully testable WITHOUT a live
control plane — see ``transport.PlaneTransport`` (the protocol), ``HttpxTransport``
(the real one), and ``tests/fake_plane.py`` (an in-memory FastAPI plane reused as
a fake). The real control plane is built by another agent; we integrate later.
"""
from .client import FederationClient, RoundResult
from .transport import HttpxTransport, PlaneTransport

__all__ = ["FederationClient", "RoundResult", "PlaneTransport", "HttpxTransport"]
