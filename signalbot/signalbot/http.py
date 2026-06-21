"""Shared, polite HTTP client.

A single place for the user-agent, timeouts, retries and inter-request delay so
every source behaves like a well-mannered bot and one knob changes them all.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import httpx

USER_AGENT = (
    "veritas-signalbot/0.1 (+market-research; respects robots; contact: set CONTACT_EMAIL)"
)


class Http:
    def __init__(
        self,
        *,
        timeout: float = 20.0,
        delay: float = 1.0,
        retries: int = 2,
        headers: Optional[dict[str, str]] = None,
    ):
        self.delay = delay
        self.retries = retries
        self._last_call = 0.0
        base_headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
        if headers:
            base_headers.update(headers)
        self._client = httpx.Client(
            timeout=timeout,
            headers=base_headers,
            follow_redirects=True,
        )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_call = time.monotonic()

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            self._throttle()
            try:
                resp = self._client.get(url, **kwargs)
                # Back off and retry on rate-limit / transient server errors.
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"status {resp.status_code}", request=resp.request, response=resp
                    )
                return resp
            except (httpx.HTTPError,) as exc:  # network + status errors
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(self.delay * (attempt + 2))
        assert last_exc is not None
        raise last_exc

    def get_json(self, url: str, **kwargs: Any) -> Any:
        resp = self.get(url, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        """Single POST (used for OAuth token exchange) — throttled, no retry so
        we never double-submit a credential grant."""
        self._throttle()
        return self._client.post(url, **kwargs)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Http":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
