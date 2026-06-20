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


def test_feature_map_resolves_snake_case_aliases(tmp_path):
    """A feature_map.yaml may declare snake_case columns (account_age) that the
    alias table resolves to the canonical contract name (accountAge)."""
    csv = tmp_path / "txns.csv"
    csv.write_text("amount,account_age,is_fraud\n1.0,-2.0,1\n0.2,0.5,0\n")
    fmap_yaml = tmp_path / "fmap.yaml"
    fmap_yaml.write_text(
        "version: 1\n"
        "source:\n  kind: csv\n  path: txns.csv\n  label_column: is_fraud\n"
        "features:\n"
        "  amount: { from: amount }\n"
        "  account_age: { from: account_age }\n"
    )
    fmap = load_feature_map(fmap_yaml)
    # stored under the canonical name, not the alias
    assert "accountAge" in fmap.features
    assert "account_age" not in fmap.features
    X, y = ConnectorRuntime(fmap).load()
    age = X[:, FEATURE_ORDER.index("accountAge")]
    assert age[0] == -2.0 and age[1] == 0.5


def test_connector_rejects_non_finite_values(tmp_path):
    """NaN/inf in a source column falls back to the default, never reaching X."""
    csv = tmp_path / "txns.csv"
    csv.write_text("amount,is_fraud\nnan,1\ninf,0\n3.0,1\n")
    fmap_yaml = tmp_path / "fmap.yaml"
    fmap_yaml.write_text(
        "version: 1\n"
        "source:\n  kind: csv\n  path: txns.csv\n  label_column: is_fraud\n"
        "features:\n  amount: { from: amount, default: 0.0 }\n"
    )
    X, _ = ConnectorRuntime(load_feature_map(fmap_yaml)).load()
    amt = X[:, FEATURE_ORDER.index("amount")]
    assert np.all(np.isfinite(amt))
    assert amt[0] == 0.0 and amt[1] == 0.0 and amt[2] == 3.0


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
