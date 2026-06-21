"""Pipeline + report tests using a stub source (no network)."""

from signalbot import pipeline as pipeline_mod
from signalbot.config import load_config
from signalbot.models import Evidence
from signalbot.pipeline import Pipeline
from signalbot.report import to_json, to_markdown
from signalbot.sources.base import Source


class StubSource(Source):
    name = "reddit"  # reuse an enabled key so the pipeline runs it

    def fetch(self):
        self.add_link("manual", "http://example/search")
        return [
            Evidence(  # strong, on-topic -> kept
                source_kind="reddit", source="r/fintech",
                title="Banks can't share fraud data on money mules",
                url="http://a",
                text="APP fraud spreads cross-bank, GDPR blocks data pooling; federated learning",
            ),
            Evidence(  # duplicate url, lower score -> deduped away
                source_kind="reddit", source="r/fintech",
                title="dupe", url="http://a", text="money mule fraud",
            ),
            Evidence(  # off-topic -> gated out
                source_kind="reddit", source="r/x",
                title="cooking recipes", url="http://b", text="no relevance",
            ),
        ]


def test_pipeline_scores_dedupes_and_sorts(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "REGISTRY", {"reddit": StubSource})
    cfg = load_config()
    cfg.queries = ["q"]
    result = Pipeline(cfg, delay=0, verbose=False).run()

    # One kept (strong), dup collapsed, off-topic dropped.
    assert len(result.evidence) == 1
    assert result.evidence[0].url == "http://a"
    assert result.counts_by_source["reddit"] == 3
    assert result.search_links and result.search_links[0].url == "http://example/search"


def test_report_renders(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "REGISTRY", {"reddit": StubSource})
    cfg = load_config()
    result = Pipeline(cfg, delay=0, verbose=False).run()

    md = to_markdown(result)
    assert "Problem-signal report" in md
    assert "Banks can't share fraud data" in md
    assert "Manual follow-up searches" in md

    js = to_json(result)
    assert '"evidence"' in js and '"problem_name"' in js
