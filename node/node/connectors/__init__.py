"""Connector runtime + declarative feature-map engine.

This is the "config not code" story: a ``feature_map.yaml`` declares how a data
source maps onto the model's FEATURE_DIM feature contract. The bank's engineers
write no code — Veritas SE authors the map and the runtime produces ``(X, y)``
numpy arrays for the local trainer.

Source kinds:
  * ``csv``       — stands in for a data warehouse (Snowflake/BigQuery/...): a CSV of transactions.
  * ``iso20022``  — parse a pacs.008 / pain.001 ISO 20022 XML message.
"""
from .feature_map import FeatureMap, load_feature_map
from .runtime import ConnectorRuntime, run_connector

__all__ = ["FeatureMap", "load_feature_map", "ConnectorRuntime", "run_connector"]
