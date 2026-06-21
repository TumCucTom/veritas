# Veritas Signal Bot

A **problem-signal scraper**. It hunts the open web for real-world evidence of
the problem Veritas claims to solve:

> Banks can't pool raw customer data to fight APP / mule fraud, so fraud
> intelligence doesn't spread across institutions — Veritas fixes this with
> privacy-preserving federated learning.

It sweeps five source types, scores every item against a configurable definition
of that problem, and emits a ranked evidence brief you can drop into a pitch
deck, investor update, or market-validation doc.

| Source | How | Auth |
|---|---|---|
| **Reddit** | public `search.json` (site-wide + per-subreddit) | none |
| **Industry forums** | Discourse `search.json` + RSS/Atom feeds | none |
| **Job descriptions** | public Greenhouse + Lever board APIs | none |
| **Conference talks** | best-effort YouTube search parse + follow-up links | none |
| **LinkedIn** | emits ready-to-click search URLs; optional cookie scrape | manual / opt-in |

LinkedIn has no public API and forbids unauthenticated scraping, so by default
the bot gives you direct search links rather than pretending to automate it. See
[`signalbot/sources/linkedin.py`](signalbot/sources/linkedin.py).

## Install

```sh
cd signalbot
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Use

```sh
signalbot run                              # all enabled sources -> markdown
signalbot run --sources reddit,hackernews,jobs
signalbot run --format json -o evidence.json
signalbot run --min-score 5 --quiet
signalbot sources                          # list source keys + enabled state
```

No console script? Use the module form: `python -m signalbot.cli run`.

## Configure — `signals.yaml`

Everything the bot looks for lives in [`signals.yaml`](signals.yaml); no code
change is needed to retarget it.

- **`problem`** — the statement you're validating (documentation only).
- **`themes`** — facets of the problem. Each has a `weight` and `phrases`;
  phrase hits add `weight` to an item's score. This is *why* an item is kept.
- **`core_terms`** — a gate: an item must contain at least one to count at all
  (keeps generic ML/privacy chatter with no fraud angle out).
- **`min_score`** — keep threshold.
- **`queries`** — search strings handed to every search-capable source.
- **`sources`** — per-source settings (subreddits, ATS companies, feeds…).
  Disable any source with `enabled: false`.

## How scoring works

Each item's `title + body` is matched (case-insensitive substring) against every
theme's phrases. First hit in a theme scores the theme `weight`; extra distinct
phrases add `+0.25` each (diminishing, so one keyword-stuffed post can't
dominate). Items must clear the `core_terms` gate and `min_score` to survive.
Cross-source duplicates (same URL) collapse to the highest-scoring copy. The
logic is intentionally transparent — see
[`signalbot/scoring.py`](signalbot/scoring.py).

## Output

Markdown brief: ranked evidence (source, score, popularity, date, matched
themes/phrases, snippet), a theme-coverage tally, and a "manual follow-up
searches" section for LinkedIn/video. JSON mirrors the full `RunResult` for
piping into other tools.

## Tests

```sh
pytest        # network-free; sources tested against fake HTTP fixtures
```

## Etiquette & limits

- One shared throttled HTTP client (`--delay`, default 1s/request) with a
  descriptive User-Agent and retry/back-off on 429/5xx.
- Only public endpoints are scraped. LinkedIn's authenticated path is opt-in
  (`use_cookie: true` + `LINKEDIN_COOKIE`) and is your compliance call.
- This is a market-research aid, not a system of record — always read the
  linked source before quoting it.
