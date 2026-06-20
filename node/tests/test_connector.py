"""Feature-map engine: csv + iso20022 sources both yield valid (X, y)."""
from pathlib import Path

import numpy as np

from veritas_core.data import FEATURE_DIM

from node.connectors import run_connector
from node.connectors.feature_map import FEATURE_ORDER, load_feature_map
from node.connectors.runtime import ConnectorRuntime

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"


def test_csv_source_produces_feature_matrix():
    X, y = run_connector(SAMPLE / "feature_map.yaml")
    assert X.shape[1] == FEATURE_DIM
    assert X.shape[0] == len(y)
    assert X.shape[0] > 100
    # labels are 0/1 and both classes present
    assert set(np.unique(y)).issubset({0, 1})
    assert y.sum() > 0 and (y == 0).sum() > 0
    # campaignSig column (last) must carry the campaign signal for mule rows
    camp_idx = FEATURE_ORDER.index("campaignSig")
    assert X[:, camp_idx].max() > 0.0


def test_csv_bool_transform():
    fmap = load_feature_map(SAMPLE / "feature_map.yaml")
    X, _ = ConnectorRuntime(fmap).load()
    is_transfer = X[:, FEATURE_ORDER.index("isTransfer")]
    assert set(np.unique(is_transfer)).issubset({0.0, 1.0})


def test_iso20022_source_parses_pacs008():
    X, y = run_connector(SAMPLE / "feature_map_iso20022.yaml")
    # two CdtTrfTxInf blocks -> two rows
    assert X.shape == (2, FEATURE_DIM)
    assert list(y) == [1, 1]
    # amount column resolved from IntrBkSttlmAmt (non-constant after zscore)
    amt = X[:, FEATURE_ORDER.index("amount")]
    assert amt.std() > 0
    # campaignSig default applied
    assert np.allclose(X[:, FEATURE_ORDER.index("campaignSig")], 1.5)
