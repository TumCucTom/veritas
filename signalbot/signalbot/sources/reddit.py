"""Reddit source.

Reddit now hard-blocks unauthenticated access to ``search.json`` (HTTP 403 from
datacenter IPs regardless of User-Agent), so the only reliable path is the
official OAuth API. This adapter:

* uses OAuth when Reddit app credentials are present in the environment, hitting
  ``oauth.reddit.com`` site-wide and per-subreddit; and
* otherwise emits ready-to-click search links (like the LinkedIn adapter) and
  returns no scraped items, rather than silently yielding zero.

Create a free "script" app at https://www.reddit.com/prefs/apps and export:

    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET      (required for OAuth)
    REDDIT_USERNAME, REDDIT_PASSWORD            (optional; enables password grant)

With username/password it uses the ``password`` grant; with only id/secret it
uses the app-only ``client_credentials`` grant.
"""

from __future__ import annotations

import os
import urllib.parse

import httpx

from .base import Source
from ..http import USER_AGENT
from ..models import Evidence

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
API = "https://oauth.reddit.com"


class RedditSource(Source):
    name = "reddit"

    def fetch(self) -> list[Evidence]:
        token = self._get_token()
        if not token:
            # No creds / auth failed — leave the analyst direct search links.
            for q in self.config.queries:
                self.add_link(
                    f"Reddit · {q}",
                    f"https://www.reddit.com/search/?q={urllib.parse.quote(q)}&sort=relevance&t=year",
                )
            return []

        headers = {"Authorization": f"bearer {token}", "User-Agent": USER_AGENT}
        out: list[Evidence] = []
        limit = int(self.cfg.get("limit_per_query", 25))
        subs = list(self.cfg.get("subreddits", []))

        for q in self.config.queries:
            out.extend(self._search(q, limit=limit, sub=None, headers=headers))
        for sub in subs:
            for q in self.config.queries:
                out.extend(self._search(q, limit=limit, sub=sub, headers=headers))
        return out

    def _get_token(self) -> str | None:
        cid = os.environ.get("REDDIT_CLIENT_ID")
        secret = os.environ.get("REDDIT_CLIENT_SECRET")
        if not cid or not secret:
            return None
        user = os.environ.get("REDDIT_USERNAME")
        pwd = os.environ.get("REDDIT_PASSWORD")
        if user and pwd:
            data = {"grant_type": "password", "username": user, "password": pwd}
        else:
            data = {"grant_type": "client_credentials"}
        try:
            resp = self.http.post(
                TOKEN_URL,
                data=data,
                auth=httpx.BasicAuth(cid, secret),
                headers={"User-Agent": USER_AGENT},
            )
            resp.raise_for_status()
            return resp.json().get("access_token")
        except Exception:
            return None

    def _search(self, query: str, *, limit: int, sub: str | None, headers: dict) -> list[Evidence]:
        params = {"q": query, "limit": str(limit), "sort": "relevance", "t": "year", "raw_json": "1"}
        if sub:
            base = f"{API}/r/{sub}/search"
            params["restrict_sr"] = "1"
        else:
            base = f"{API}/search"
        url = f"{base}?{urllib.parse.urlencode(params)}"

        try:
            data = self.http.get_json(url, headers=headers)
        except Exception:
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
