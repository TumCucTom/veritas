"""Transport seam between the federation client and the control plane.

``PlaneTransport`` is the narrow interface the client depends on — every method
maps 1:1 to a PROTOCOL.md Tier 2 endpoint. ``HttpxTransport`` is the production
implementation (outbound-only TLS via httpx). Tests inject a fake that talks to
an in-memory plane instead, so federation rounds run with no network.
"""
from __future__ import annotations

from typing import Any, Protocol

import httpx


class PlaneTransport(Protocol):
    # --- enrolment (unauthenticated; establishes identity) ---
    def enroll(self, body: dict[str, Any]) -> dict[str, Any]: ...

    # --- federation rounds ---
    def get_current_round(self) -> dict[str, Any]: ...
    def get_current_model(self) -> dict[str, Any]: ...
    def submit_update(self, round_no: int, body: dict[str, Any], token: str) -> dict[str, Any]: ...
    def get_round_result(self, round_no: int) -> dict[str, Any]: ...


class HttpxTransport:
    """Real transport: JSON over HTTP to the control plane (outbound-only)."""

    def __init__(self, base_url: str, *, timeout: float = 10.0, client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def enroll(self, body: dict[str, Any]) -> dict[str, Any]:
        r = self._client.post(self._url("/v1/members/enroll"), json=body)
        r.raise_for_status()
        return r.json()

    def get_current_round(self) -> dict[str, Any]:
        r = self._client.get(self._url("/v1/rounds/current"))
        r.raise_for_status()
        return r.json()

    def get_current_model(self) -> dict[str, Any]:
        r = self._client.get(self._url("/v1/models/current"))
        r.raise_for_status()
        return r.json()

    def submit_update(self, round_no: int, body: dict[str, Any], token: str) -> dict[str, Any]:
        r = self._client.post(
            self._url(f"/v1/rounds/{round_no}/updates"),
            json=body,
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()

    def get_round_result(self, round_no: int) -> dict[str, Any]:
        r = self._client.get(self._url(f"/v1/rounds/{round_no}/result"))
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._client.close()
