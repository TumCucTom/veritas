from signalbot.config import load_config
from signalbot.models import Evidence
from signalbot.scoring import Scorer


def _scorer():
    return Scorer(load_config())


def test_on_topic_item_scores_and_tags_themes():
    s = _scorer()
    ev = Evidence(
        source_kind="reddit",
        source="r/fintech",
        title="Banks can't share fraud data to stop money mules",
        text="We see APP fraud move cross-bank but GDPR blocks data pooling. "
        "Federated learning could help.",
    )
    s.score(ev)
    assert ev.score >= s.config.min_score
    assert "data_sharing_barrier" in ev.themes
    assert "app_fraud" in ev.themes
    assert "mule_accounts" in ev.themes
    assert s.keep(ev)


def test_core_term_gate_rejects_unrelated_ml_chatter():
    s = _scorer()
    ev = Evidence(
        source_kind="hackernews",
        source="hn",
        title="Federated learning and differential privacy for ad targeting",
        text="privacy preserving on-device model for advertising, totally unrelated topic",
    )
    s.score(ev)
    # Matches privacy theme phrases but no core fraud term -> gated to 0.
    assert ev.score == 0.0
    assert not s.keep(ev)


def test_weak_single_theme_below_threshold():
    s = _scorer()
    ev = Evidence(
        source_kind="forum",
        source="x",
        title="Generic fraud detection tips",
        text="some fraud detection advice",
    )
    s.score(ev)
    # Only the weight-1 fraud_detection theme -> below default min_score (3).
    assert ev.score < s.config.min_score
