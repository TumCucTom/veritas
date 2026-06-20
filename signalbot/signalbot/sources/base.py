"""Source adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import Config
from ..http import Http
from ..models import Evidence, SearchLink


class Source(ABC):
    """One scrapeable origin (Reddit, HN, jobs...).

    A source is responsible only for *fetching candidate items* and returning
    them as raw ``Evidence``. Scoring, de-duplication and filtering happen later
    in the pipeline, so a source never decides relevance itself.
    """

    #: stable key, also the key under `sources:` in signals.yaml
    name: str = "base"

    def __init__(self, config: Config, http: Http):
        self.config = config
        self.http = http
        self.cfg = config.source_cfg(self.name)
        # Side-channel: manual follow-up links a source wants surfaced.
        self.search_links: list[SearchLink] = []

    @abstractmethod
    def fetch(self) -> list[Evidence]:
        """Return candidate evidence. Must not raise on a single bad query —
        log/collect partial results and continue."""

    # Convenience for emitting a manual search URL.
    def add_link(self, label: str, url: str) -> None:
        self.search_links.append(
            SearchLink(source_kind=self.name, label=label, url=url)
        )
