"""Industry-forum source: Discourse search APIs + plain RSS/Atom feeds.

Two cheap, robust mechanisms that cover most fintech / fraud / banking forums:

* **Discourse** communities (very common platform) expose ``/search.json?q=``.
* Everything else usually publishes an **RSS/Atom** feed; we pull recent items
  and let the scorer decide relevance (feeds aren't query-able, so this is a
  "recent activity" sweep rather than a targeted search).
"""

from __future__ import annotations

import re
import urllib.parse
from xml.etree import ElementTree as ET

from .base import Source
from ..models import Evidence

_TAG = re.compile(r"<[^>]+>")


def _strip_html(s: str) -> str:
    return _TAG.sub(" ", s or "").strip()


class ForumsSource(Source):
    name = "forums"

    def fetch(self) -> list[Evidence]:
        out: list[Evidence] = []
        for base in self.cfg.get("discourse", []) or []:
            out.extend(self._discourse(base.rstrip("/")))
        for feed in self.cfg.get("rss", []) or []:
            out.extend(self._rss(feed))
        return out

    # --- Discourse ---------------------------------------------------------
    def _discourse(self, base: str) -> list[Evidence]:
        evidence: list[Evidence] = []
        for q in self.config.queries:
            url = f"{base}/search.json?{urllib.parse.urlencode({'q': q})}"
            try:
                data = self.http.get_json(url)
            except Exception:
                self.add_link(f"Search {base}", f"{base}/search?q={urllib.parse.quote(q)}")
                continue
            topics = {t["id"]: t for t in data.get("topics", [])}
            for post in data.get("posts", []):
                topic = topics.get(post.get("topic_id"), {})
                slug = topic.get("slug", "")
                tid = post.get("topic_id", "")
                evidence.append(
                    Evidence(
                        source_kind="forum",
                        source=urllib.parse.urlparse(base).netloc,
                        title=topic.get("title", "") or "",
                        url=f"{base}/t/{slug}/{tid}" if slug else base,
                        author=post.get("username"),
                        text=_strip_html(post.get("blurb", ""))[:4000],
                    )
                )
        return evidence

    # --- RSS / Atom --------------------------------------------------------
    def _rss(self, feed_url: str) -> list[Evidence]:
        try:
            resp = self.http.get(feed_url)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception:
            self.add_link("Open feed", feed_url)
            return []

        netloc = urllib.parse.urlparse(feed_url).netloc
        evidence: list[Evidence] = []

        # RSS 2.0: channel/item ; Atom: feed/entry
        items = root.findall(".//item") or root.findall(
            ".//{http://www.w3.org/2005/Atom}entry"
        )
        for it in items:
            title = _first_text(it, ["title", "{http://www.w3.org/2005/Atom}title"])
            link = _first_text(it, ["link", "guid"]) or _atom_link(it)
            desc = _first_text(
                it,
                [
                    "description",
                    "{http://www.w3.org/2005/Atom}summary",
                    "{http://www.w3.org/2005/Atom}content",
                ],
            )
            evidence.append(
                Evidence(
                    source_kind="forum",
                    source=netloc,
                    title=title,
                    url=link,
                    text=_strip_html(desc)[:4000],
                )
            )
        return evidence


def _first_text(el: ET.Element, tags: list[str]) -> str:
    for tag in tags:
        found = el.find(tag)
        if found is not None and (found.text or "").strip():
            return found.text.strip()
    return ""


def _atom_link(el: ET.Element) -> str:
    link = el.find("{http://www.w3.org/2005/Atom}link")
    if link is not None:
        return link.get("href", "")
    return ""
