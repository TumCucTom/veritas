"""Source-adapter parsing tests with a fake HTTP client (no network)."""

from signalbot.config import load_config
from signalbot.sources.hackernews import HackerNewsSource
from signalbot.sources.jobs import JobsSource
from signalbot.sources.reddit import RedditSource


class FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class FakeHttp:
    """Returns a payload chosen by substring match on the requested URL."""

    def __init__(self, routes: dict[str, object]):
        self.routes = routes
        self.calls: list[str] = []

    def _match(self, url):
        for needle, payload in self.routes.items():
            if needle in url:
                return payload
        return None

    def get_json(self, url, **kwargs):
        self.calls.append(url)
        payload = self._match(url)
        if payload is None:
            raise RuntimeError(f"no route for {url}")
        return payload

    def get(self, url, **kwargs):
        self.calls.append(url)
        payload = self._match(url)
        if payload is None:
            raise RuntimeError(f"no route for {url}")
        return FakeResp(payload)


def _config_one_query():
    cfg = load_config()
    cfg.queries = ["fraud data sharing"]
    return cfg


def test_reddit_parses_children():
    cfg = _config_one_query()
    cfg.sources["reddit"] = {"enabled": True, "subreddits": [], "limit_per_query": 5}
    http = FakeHttp({
        "search.json": {
            "data": {
                "children": [
                    {"data": {
                        "subreddit": "fintech",
                        "title": "Banks cannot share fraud data",
                        "permalink": "/r/fintech/abc",
                        "author": "alice",
                        "created_utc": 1700000000,
                        "selftext": "GDPR blocks pooling customer data",
                        "score": 42,
                    }}
                ]
            }
        }
    })
    items = RedditSource(cfg, http).fetch()
    assert len(items) == 1
    ev = items[0]
    assert ev.source == "r/fintech"
    assert ev.url == "https://www.reddit.com/r/fintech/abc"
    assert ev.popularity == 42


def test_hackernews_parses_story_and_comment():
    cfg = _config_one_query()
    http = FakeHttp({
        "hn.algolia.com": {
            "hits": [
                {"objectID": "1", "title": "Fraud sharing", "url": "http://x",
                 "author": "bob", "created_at_i": 1700000000, "points": 10},
                {"objectID": "2", "comment_text": "we cannot pool data",
                 "story_title": "thread", "author": "carol", "num_comments": 3},
            ]
        }
    })
    items = HackerNewsSource(cfg, http).fetch()
    assert len(items) == 2
    assert items[0].url == "http://x"
    # comment falls back to HN item permalink
    assert "news.ycombinator.com/item?id=2" in items[1].url


def test_jobs_greenhouse_keyword_filter():
    cfg = _config_one_query()
    cfg.sources["jobs"] = {
        "enabled": True,
        "greenhouse": ["acme"],
        "lever": [],
        "keywords_any": ["fraud"],
    }
    http = FakeHttp({
        "greenhouse.io": {
            "jobs": [
                {"title": "Fraud ML Engineer", "absolute_url": "http://j/1",
                 "content": "Build cross-bank &lt;b&gt;fraud&lt;/b&gt; detection"},
                {"title": "Frontend Engineer", "absolute_url": "http://j/2",
                 "content": "React work, nothing relevant"},
            ]
        }
    })
    items = JobsSource(cfg, http).fetch()
    # Only the fraud posting passes the keyword filter.
    assert len(items) == 1
    assert items[0].title == "Fraud ML Engineer"
    # entities decoded then tags stripped -> readable text, no markup left
    assert "fraud" in items[0].text.lower()
    assert "&lt;" not in items[0].text and "<b>" not in items[0].text


def test_source_failure_returns_empty_not_raise():
    cfg = _config_one_query()
    cfg.sources["reddit"] = {"enabled": True, "subreddits": [], "limit_per_query": 5}
    http = FakeHttp({})  # every URL 404s in the fake
    items = RedditSource(cfg, http).fetch()
    assert items == []
