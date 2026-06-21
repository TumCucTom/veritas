"""Relevance scoring: turn raw text into a problem-relevance score + matched themes.

Deliberately simple and explainable (weighted phrase hits) rather than an opaque
model — an analyst can see *why* an item was kept. The scorer is the one place
that decides whether a scraped item is "evidence of the problem".
"""

from __future__ import annotations

from .config import Config
from .models import Evidence


class Scorer:
    def __init__(self, config: Config):
        self.config = config
        # Pre-lower phrases once.
        self._themes = {
            name: (theme.weight, [p.lower() for p in theme.phrases])
            for name, theme in config.themes.items()
        }
        self._core = [c.lower() for c in config.core_terms]

    def score(self, ev: Evidence) -> Evidence:
        """Annotate ``ev`` in place with score / themes / matched_phrases."""
        hay = ev.haystack()

        # Gate: must contain at least one core term to be on-topic at all.
        if self._core and not any(term in hay for term in self._core):
            ev.score = 0.0
            return ev

        total = 0.0
        matched_themes: list[str] = []
        matched_phrases: list[str] = []

        for name, (weight, phrases) in self._themes.items():
            hits = [p for p in phrases if p in hay]
            if hits:
                matched_themes.append(name)
                matched_phrases.extend(hits)
                # First hit in a theme is worth the theme weight; additional
                # distinct phrases add diminishing value so one keyword-stuffed
                # post can't dominate.
                total += weight + 0.25 * (len(hits) - 1)

        ev.score = round(total, 2)
        ev.themes = matched_themes
        ev.matched_phrases = sorted(set(matched_phrases))
        return ev

    def keep(self, ev: Evidence) -> bool:
        return ev.score >= self.config.min_score
