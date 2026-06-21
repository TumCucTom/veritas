"""Orchestrator: run sources, score, de-duplicate, sort."""

from __future__ import annotations

import sys

from .config import Config
from .http import Http
from .models import Evidence, RunResult, SearchLink
from .scoring import Scorer
from .sources import REGISTRY


class Pipeline:
    def __init__(self, config: Config, *, delay: float = 1.0, verbose: bool = True):
        self.config = config
        self.scorer = Scorer(config)
        self.delay = delay
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)

    def run(self, only: list[str] | None = None) -> RunResult:
        result = RunResult(
            problem_name=self.config.problem.name,
            queries=list(self.config.queries),
        )
        raw: list[Evidence] = []
        links: list[SearchLink] = []

        with Http(delay=self.delay) as http:
            for name, cls in REGISTRY.items():
                if only and name not in only:
                    continue
                if not self.config.source_enabled(name):
                    self._log(f"· {name}: disabled, skipping")
                    continue
                self._log(f"→ {name}: fetching…")
                source = cls(self.config, http)
                try:
                    items = source.fetch()
                except Exception as exc:  # a source must never sink the run
                    result.errors.append(f"{name}: {exc!r}")
                    self._log(f"  ! {name} failed: {exc!r}")
                    items = []
                raw.extend(items)
                links.extend(source.search_links)
                result.counts_by_source[name] = result.counts_by_source.get(name, 0) + len(items)
                self._log(f"  {name}: {len(items)} candidate(s)")

        # Score, gate, de-duplicate (keep highest-scoring per key).
        best: dict[str, Evidence] = {}
        for ev in raw:
            self.scorer.score(ev)
            if not self.scorer.keep(ev):
                continue
            key = ev.dedupe_key
            if key not in best or ev.score > best[key].score:
                best[key] = ev

        evidence = sorted(best.values(), key=lambda e: e.score, reverse=True)
        result.evidence = evidence
        result.search_links = links
        self._log(f"✓ kept {len(evidence)} evidence item(s) above score {self.config.min_score}")
        return result
