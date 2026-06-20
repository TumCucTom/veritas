"""Reddit source — public JSON search, no API key required.

Reddit exposes ``/search.json`` and ``/r/<sub>/search.json`` without auth (rate
limited). We run every configured query against the whole site and against each
listed subreddit, then hand raw posts to the scorer.
"""

from __future__ import annotations

import urllib.parse

from .base import Source
from ..models import Evidence


class RedditSource(Source):
    name = "reddit"

    def fetch(self) -> list[Evidence]:
        out: list[Evidence] = []
        limit = int(self.cfg.get("limit_per_query", 25))
        subs = list(self.cfg.get("subreddits", []))
        queries = self.config.queries

        # Site-wide searches.
        for q in queries:
            out.extend(self._search(q, limit=limit, sub=None))
        # Targeted subreddit searches (restrict_sr keeps results on-topic).
        for sub in subs:
            for q in queries:
                out.extend(self._search(q, limit=limit, sub=sub))

        return out

    def _search(self, query: str, *, limit: int, sub: str | None) -> list[Evidence]:
        params = {
            "q": query,
            "limit": str(limit),
            "sort": "relevance",
            "t": "year",
            "raw_json": "1",
        }
        if sub:
            base = f"https://www.reddit.com/r/{sub}/search.json"
            params["restrict_sr"] = "1"
        else:
            base = "https://www.reddit.com/search.json"
        url = f"{base}?{urllib.parse.urlencode(params)}"

        try:
            data = self.http.get_json(url)
        except Exception:
            # One failed query shouldn't sink the source.
            return []

        evidence: list[Evidence] = []
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            permalink = d.get("permalink", "")
            evidence.append(
                Evidence(
                    source_kind="reddit",
                    source=f"r/{d.get('subreddit', sub or '?')}",
                    title=d.get("title", "") or "",
                    url=f"https://www.reddit.com{permalink}" if permalink else d.get("url", ""),
                    author=d.get("author"),
                    created_utc=d.get("created_utc"),
                    text=(d.get("selftext", "") or "")[:4000],
                    popularity=d.get("score"),
                )
            )
        return evidence
