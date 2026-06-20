"""LinkedIn source — honest about the wall.

LinkedIn has no public search API and its terms forbid unauthenticated scraping;
anything else would be brittle and a compliance risk. So by default this adapter
produces **ready-to-click search URLs** (content + jobs) for each query — the
analyst stays in the loop for the one source that can't be safely automated.

If you accept the ToS/risk and export ``LINKEDIN_COOKIE`` (your ``li_at`` cookie)
*and* set ``use_cookie: true`` in signals.yaml, it makes a single best-effort
call to the authenticated Voyager blended-search endpoint and degrades silently
to links on any change/failure. Treat that path as experimental.
"""

from __future__ import annotations

import os
import urllib.parse

from .base import Source
from ..models import Evidence


class LinkedInSource(Source):
    name = "linkedin"

    def fetch(self) -> list[Evidence]:
        evidence: list[Evidence] = []

        # Always emit manual search links — the reliable path.
        for q in self.config.queries:
            enc = urllib.parse.quote(q)
            self.add_link(
                f"Posts · {q}",
                f"https://www.linkedin.com/search/results/content/?keywords={enc}&sortBy=%22date_posted%22",
            )
            self.add_link(
                f"Jobs · {q}",
                f"https://www.linkedin.com/jobs/search/?keywords={enc}",
            )

        # Opt-in authenticated best-effort.
        if self.cfg.get("use_cookie") and os.environ.get("LINKEDIN_COOKIE"):
            evidence.extend(self._voyager_best_effort())

        return evidence

    def _voyager_best_effort(self) -> list[Evidence]:
        cookie = os.environ["LINKEDIN_COOKIE"].strip()
        headers = {
            "Cookie": f"li_at={cookie}",
            "Csrf-Token": "ajax:0000000000000000000",
            "X-RestLi-Protocol-Version": "2.0.0",
            "Accept": "application/json",
        }
        out: list[Evidence] = []
        for q in self.config.queries:
            params = {
                "keywords": q,
                "origin": "GLOBAL_SEARCH_HEADER",
                "q": "all",
            }
            url = (
                "https://www.linkedin.com/voyager/api/search/blended?"
                + urllib.parse.urlencode(params)
            )
            try:
                data = self.http.get_json(url, headers=headers)
            except Exception:
                continue  # links already cover us
            for el in _iter_voyager_updates(data):
                out.append(
                    Evidence(
                        source_kind="linkedin",
                        source="linkedin.com",
                        title=el.get("title", "")[:200],
                        url=el.get("url", ""),
                        text=el.get("text", "")[:4000],
                    )
                )
        return out


def _iter_voyager_updates(data: dict) -> list[dict]:
    """Voyager responses are deeply nested and change often; pull anything that
    looks like a post with text, defensively."""
    results: list[dict] = []
    included = data.get("included", []) if isinstance(data, dict) else []
    for item in included:
        if not isinstance(item, dict):
            continue
        commentary = item.get("commentary") or {}
        text = ""
        if isinstance(commentary, dict):
            text = (commentary.get("text") or {}).get("text", "")
        if text:
            results.append(
                {
                    "title": text[:120],
                    "text": text,
                    "url": item.get("permalink") or item.get("url") or "",
                }
            )
    return results
