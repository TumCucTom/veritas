"""Render a RunResult as JSON or a human-readable Markdown brief."""

from __future__ import annotations

import datetime as _dt
import json

from .models import RunResult


def to_json(result: RunResult) -> str:
    return json.dumps(result.model_dump(), indent=2, ensure_ascii=False)


def _ts(epoch: float | None) -> str:
    if not epoch:
        return ""
    try:
        return _dt.datetime.fromtimestamp(epoch, _dt.timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return ""


def to_markdown(result: RunResult) -> str:
    lines: list[str] = []
    lines.append(f"# Problem-signal report")
    lines.append("")
    lines.append(f"**Problem:** {result.problem_name}")
    lines.append("")
    lines.append(f"**Evidence items kept:** {len(result.evidence)}")
    lines.append("")

    # Source tally.
    if result.counts_by_source:
        lines.append("**Candidates fetched by source:** " + ", ".join(
            f"{k} ({v})" for k, v in sorted(result.counts_by_source.items())
        ))
        lines.append("")

    # Theme tally across kept evidence — shows which facet of the problem the
    # market is actually talking about.
    theme_counts: dict[str, int] = {}
    for ev in result.evidence:
        for t in ev.themes:
            theme_counts[t] = theme_counts.get(t, 0) + 1
    if theme_counts:
        lines.append("**Theme coverage (kept items):**")
        for t, c in sorted(theme_counts.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"- `{t}`: {c}")
        lines.append("")

    lines.append("---")
    lines.append("## Evidence (ranked by relevance)")
    lines.append("")
    if not result.evidence:
        lines.append("_No items cleared the score threshold. Try lowering `min_score` "
                     "or broadening `queries` in signals.yaml._")
    for i, ev in enumerate(result.evidence, 1):
        date = _ts(ev.created_utc)
        meta = " · ".join(
            x for x in [
                ev.source,
                f"score {ev.score}",
                f"▲{ev.popularity}" if ev.popularity is not None else "",
                date,
            ] if x
        )
        title = ev.title or "(untitled)"
        lines.append(f"### {i}. {title}")
        lines.append(f"{meta}")
        if ev.url:
            lines.append(f"<{ev.url}>")
        if ev.themes:
            lines.append(f"_themes:_ {', '.join(ev.themes)}")
        if ev.matched_phrases:
            lines.append(f"_matched:_ {', '.join(ev.matched_phrases)}")
        snippet = (ev.text or "").strip().replace("\n", " ")
        if snippet:
            lines.append("")
            lines.append("> " + (snippet[:300] + ("…" if len(snippet) > 300 else "")))
        lines.append("")

    if result.search_links:
        lines.append("---")
        lines.append("## Manual follow-up searches")
        lines.append("")
        lines.append("_Sources that can't be safely auto-scraped (LinkedIn) or are "
                     "fragile (video search) leave you these direct links:_")
        lines.append("")
        by_kind: dict[str, list] = {}
        for link in result.search_links:
            by_kind.setdefault(link.source_kind, []).append(link)
        for kind, links in by_kind.items():
            lines.append(f"**{kind}**")
            for link in links[:40]:
                lines.append(f"- [{link.label}]({link.url})")
            lines.append("")

    if result.errors:
        lines.append("---")
        lines.append("## Errors")
        for e in result.errors:
            lines.append(f"- {e}")
        lines.append("")

    return "\n".join(lines)
