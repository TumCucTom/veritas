"""Typed data structures shared across the bot."""

from __future__ import annotations

import hashlib
from typing import Optional

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A single scraped item that may be evidence of the target problem.

    Sources produce ``Evidence`` with ``score``/``themes`` unset; the scorer
    fills those in. Items below ``min_score`` are dropped by the pipeline.
    """

    source_kind: str = Field(..., description="reddit | hackernews | forum | job | linkedin | conftalk")
    source: str = Field(..., description="Concrete origin, e.g. 'r/fintech' or 'greenhouse:monzo'")
    title: str = ""
    url: str = ""
    author: Optional[str] = None
    created_utc: Optional[float] = None
    text: str = Field("", description="Body / snippet used for scoring")
    # Source-native popularity signal (upvotes, points, comment count...).
    popularity: Optional[int] = None

    # Filled in by the scorer.
    score: float = 0.0
    themes: list[str] = Field(default_factory=list)
    matched_phrases: list[str] = Field(default_factory=list)

    @property
    def dedupe_key(self) -> str:
        """Stable identity for cross-source de-duplication."""
        basis = (self.url or f"{self.source}:{self.title}").strip().lower()
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()

    def haystack(self) -> str:
        return f"{self.title}\n{self.text}".lower()


class SearchLink(BaseModel):
    """A manual-follow-up search URL emitted by sources that can't be fully
    scraped without authentication (LinkedIn) or that are inherently fragile
    (conference video search). Keeps the analyst's next click one tap away."""

    source_kind: str
    label: str
    url: str


class RunResult(BaseModel):
    """Everything one invocation produced."""

    problem_name: str
    queries: list[str]
    evidence: list[Evidence] = Field(default_factory=list)
    search_links: list[SearchLink] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    counts_by_source: dict[str, int] = Field(default_factory=dict)
