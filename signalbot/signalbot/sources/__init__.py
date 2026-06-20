"""Source registry — maps config keys to adapter classes."""

from __future__ import annotations

from .base import Source
from .conftalks import ConfTalksSource
from .forums import ForumsSource
from .hackernews import HackerNewsSource
from .jobs import JobsSource
from .linkedin import LinkedInSource
from .reddit import RedditSource

REGISTRY: dict[str, type[Source]] = {
    RedditSource.name: RedditSource,
    HackerNewsSource.name: HackerNewsSource,
    ForumsSource.name: ForumsSource,
    JobsSource.name: JobsSource,
    LinkedInSource.name: LinkedInSource,
    ConfTalksSource.name: ConfTalksSource,
}

__all__ = ["REGISTRY", "Source"]
