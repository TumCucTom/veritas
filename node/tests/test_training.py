"""Local training on connector data improves recall over the untrained model."""
from pathlib import Path

from veritas_core.data import FEATURE_DIM
from veritas_core.model import init_weights, recall, train_local

from node.connectors import run_connector

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"


def test_local_training_improves_recall():
    X, y = run_connector(SAMPLE / "feature_map.yaml")
    w0 = init_weights(FEATURE_DIM)
    r0 = recall(w0, X, y)
    w1 = train_local(w0, X, y, epochs=8, lr=0.3)
    r1 = recall(w1, X, y)
    assert r1 > r0
    assert r1 > 0.5
