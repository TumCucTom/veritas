"""Hacker News source via the Algolia search API (no key).

HN stands in for "industry / practitioner forum": founders, fraud engineers and
fintech folk discussing exactly the data-sharing pain Veritas targets. The
Algolia endpoint returns stories *and* comments — comments are often where the
candid "we literally can't share this data" admissions live.
"""

from __future__ import annotations

import urllib.parse

from .base import Source
from ..models import Evidence

ALGOLIA = "https://hn.algolia.com/api/v1/search"


class HackerNewsSource(Source):
    name = "hackernews"

    def fetch(self) -> list[Evidence]:
        out: list[Evidence] = []
        limit = int(self.cfg.get("limit_per_query", 25))
        for q in self.config.queries:
            out.extend(self._search(q, limit))
        return out

    def _search(self, query: str, limit: int) -> list[Evidence]:
        params = {
            "query": query,
            "tags": "(story,comment)",
            "hitsPerPage": str(limit),
        }
        url = f"{ALGOLIA}?{urllib.parse.urlencode(params)}"
        try:
            data = self.http.get_json(url)
        except Exception:
            return []

        evidence: list[Evidence] = []
        for hit in data.get("hits", []):
            object_id = hit.get("objectID")
            title = hit.get("title") or hit.get("story_title") or "(comment)"
            text = hit.get("comment_text") or hit.get("story_text") or ""
            url_ = hit.get("url") or (
                f"https://news.ycombinator.com/item?id={object_id}" if object_id else ""
            )
            evidence.append(
                Evidence(
                    source_kind="hackernews",
                    source="news.ycombinator.com",
                    title=title,
                    url=url_,
                    author=hit.get("author"),
                    created_utc=hit.get("created_at_i"),
                    text=text[:4000],
                    popularity=hit.get("points") or hit.get("num_comments"),
                )
            )
        return evidence
