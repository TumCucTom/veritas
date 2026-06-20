"""Conference-talk source — best-effort YouTube search (no API key).

Conference talks live mostly on YouTube. The results page embeds a JSON blob
(``ytInitialData``) we can parse without an API key. It's inherently fragile
(YouTube changes markup), so on any failure we fall back to emitting search URLs
for each query. We bias queries toward talk-shaped phrasing ("... conference
talk") so hits skew toward sessions rather than news clips.
"""

from __future__ import annotations

import json
import re
import urllib.parse

from .base import Source
from ..models import Evidence

_INITIAL = re.compile(r"var ytInitialData = (\{.*?\});", re.DOTALL)


class ConfTalksSource(Source):
    name = "conftalks"

    def fetch(self) -> list[Evidence]:
        out: list[Evidence] = []
        for q in self.config.queries:
            talk_q = f"{q} conference talk"
            enc = urllib.parse.quote(talk_q)
            self.add_link(
                f"YouTube · {talk_q}",
                f"https://www.youtube.com/results?search_query={enc}",
            )
            if self.cfg.get("youtube", True):
                out.extend(self._youtube(talk_q))
        return out

    def _youtube(self, query: str) -> list[Evidence]:
        enc = urllib.parse.quote(query)
        url = f"https://www.youtube.com/results?search_query={enc}"
        try:
            resp = self.http.get(url, headers={"Accept-Language": "en-US,en;q=0.9"})
            resp.raise_for_status()
            m = _INITIAL.search(resp.text)
            if not m:
                return []
            data = json.loads(m.group(1))
        except Exception:
            return []

        out: list[Evidence] = []
        for vid in _iter_video_renderers(data):
            video_id = vid.get("videoId")
            if not video_id:
                continue
            title = _runs_text(vid.get("title", {}))
            desc = _runs_text(vid.get("detailedMetadataSnippets", [{}]))
            channel = _runs_text(vid.get("ownerText", {}))
            out.append(
                Evidence(
                    source_kind="conftalk",
                    source=f"youtube:{channel}" if channel else "youtube",
                    title=title,
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    author=channel or None,
                    text=desc[:2000],
                )
            )
        return out


def _iter_video_renderers(data: dict) -> list[dict]:
    """Walk the ytInitialData tree and yield every videoRenderer dict."""
    found: list[dict] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if "videoRenderer" in node and isinstance(node["videoRenderer"], dict):
                found.append(node["videoRenderer"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return found


def _runs_text(node: object) -> str:
    """Flatten YouTube's {'runs':[{'text':...}]} or {'simpleText':...} shapes."""
    if isinstance(node, list):
        return " ".join(_runs_text(n) for n in node).strip()
    if isinstance(node, dict):
        if "simpleText" in node:
            return str(node["simpleText"])
        if "runs" in node:
            return "".join(r.get("text", "") for r in node["runs"])
        if "snippetText" in node:
            return _runs_text(node["snippetText"])
    return ""
