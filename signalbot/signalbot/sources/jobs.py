"""Job-description source: public Greenhouse + Lever boards (no API key).

A company hiring "Fraud ML Engineer — build cross-bank detection without sharing
PII" is hard evidence the problem is real and funded. Both ATS platforms expose
public JSON board endpoints, so we pull every posting and keep those that mention
a configured fraud/privacy keyword; the scorer then ranks them like any other
item.
"""

from __future__ import annotations

import re

from .base import Source
from ..models import Evidence

_TAG = re.compile(r"<[^>]+>")
GREENHOUSE = "https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true"
LEVER = "https://api.lever.co/v0/postings/{company}?mode=json"


def _strip_html(s: str) -> str:
    # Greenhouse returns HTML-escaped content; a light strip is enough for
    # keyword scoring (we don't render it).
    s = s.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return _TAG.sub(" ", s or "")


class JobsSource(Source):
    name = "jobs"

    def fetch(self) -> list[Evidence]:
        out: list[Evidence] = []
        keywords = [k.lower() for k in self.cfg.get("keywords_any", [])]
        for company in self.cfg.get("greenhouse", []) or []:
            out.extend(self._greenhouse(company, keywords))
        for company in self.cfg.get("lever", []) or []:
            out.extend(self._lever(company, keywords))
        return out

    def _relevant(self, text: str, keywords: list[str]) -> bool:
        if not keywords:
            return True
        low = text.lower()
        return any(k in low for k in keywords)

    def _greenhouse(self, company: str, keywords: list[str]) -> list[Evidence]:
        try:
            data = self.http.get_json(GREENHOUSE.format(company=company))
        except Exception:
            self.add_link(
                f"Greenhouse: {company}", f"https://boards.greenhouse.io/{company}"
            )
            return []
        out: list[Evidence] = []
        for job in data.get("jobs", []):
            content = _strip_html(job.get("content", ""))
            title = job.get("title", "") or ""
            if not self._relevant(f"{title} {content}", keywords):
                continue
            out.append(
                Evidence(
                    source_kind="job",
                    source=f"greenhouse:{company}",
                    title=title,
                    url=job.get("absolute_url", ""),
                    text=content[:5000],
                )
            )
        return out

    def _lever(self, company: str, keywords: list[str]) -> list[Evidence]:
        try:
            data = self.http.get_json(LEVER.format(company=company))
        except Exception:
            self.add_link(f"Lever: {company}", f"https://jobs.lever.co/{company}")
            return []
        out: list[Evidence] = []
        for job in data if isinstance(data, list) else []:
            content = _strip_html(job.get("descriptionPlain") or job.get("description", ""))
            title = job.get("text", "") or ""
            if not self._relevant(f"{title} {content}", keywords):
                continue
            out.append(
                Evidence(
                    source_kind="job",
                    source=f"lever:{company}",
                    title=title,
                    url=job.get("hostedUrl", ""),
                    created_utc=(job.get("createdAt") or 0) / 1000 or None,
                    text=content[:5000],
                )
            )
        return out
