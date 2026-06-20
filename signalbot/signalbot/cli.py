"""Command-line entrypoint.

    signalbot run                          # all enabled sources, markdown to stdout
    signalbot run --sources reddit,jobs    # only these
    signalbot run --format json -o out.json
    signalbot run --min-score 5 --config my_signals.yaml
    signalbot sources                      # list source keys + enabled state
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .pipeline import Pipeline
from .report import to_json, to_markdown
from .sources import REGISTRY


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.min_score is not None:
        config.min_score = args.min_score

    only = None
    if args.sources:
        only = [s.strip() for s in args.sources.split(",") if s.strip()]
        unknown = [s for s in only if s not in REGISTRY]
        if unknown:
            print(f"unknown source(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"valid: {', '.join(REGISTRY)}", file=sys.stderr)
            return 2

    pipeline = Pipeline(config, delay=args.delay, verbose=not args.quiet)
    result = pipeline.run(only=only)

    rendered = to_json(result) if args.format == "json" else to_markdown(result)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output} ({len(result.evidence)} evidence items)", file=sys.stderr)
    else:
        print(rendered)
    return 0


def _cmd_sources(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    for name in REGISTRY:
        state = "enabled" if config.source_enabled(name) else "disabled"
        print(f"{name:<12} {state}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="signalbot",
        description="Scrape Reddit, forums, LinkedIn, conference talks and job "
        "descriptions for evidence of the problem defined in signals.yaml.",
    )
    p.add_argument("--config", help="path to signals.yaml (default: bundled)")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the scrape and emit a report")
    run.add_argument("--sources", help="comma-separated subset, e.g. reddit,jobs")
    run.add_argument("--format", choices=["markdown", "json"], default="markdown")
    run.add_argument("-o", "--output", help="write report to file instead of stdout")
    run.add_argument("--min-score", type=float, help="override min_score from config")
    run.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    run.add_argument("--quiet", action="store_true", help="suppress progress on stderr")
    run.set_defaults(func=_cmd_run)

    src = sub.add_parser("sources", help="list source keys and enabled state")
    src.set_defaults(func=_cmd_sources)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
