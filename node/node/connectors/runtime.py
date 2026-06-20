"""Connector runtime: source readers + feature-map resolution → (X, y).

The runtime reads raw records from a source (csv warehouse stand-in or ISO 20022
XML), then applies the declarative feature map to project each record onto the
FEATURE_DIM feature contract. The output is exactly the ``(X, y)`` pair that
``veritas_core.model.train_local`` / ``recall`` consume.

No customer data leaves this process — the runtime runs in-tenancy on the node.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from veritas_core.data import FEATURE_DIM

from .feature_map import FEATURE_ORDER, FeatureMap, FeatureSpec, load_feature_map


def _finite_or_default(x: float, default: float) -> float:
    """Reject NaN/inf parsed from a source: a non-finite feature would poison
    training/DP/aggregation downstream, so we fall back to the column default
    rather than letting it through."""
    return x if math.isfinite(x) else default


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return _finite_or_default(float(value), default)
    s = str(value).strip()
    if s == "":
        return default
    low = s.lower()
    if low in ("true", "yes", "y"):
        return 1.0
    if low in ("false", "no", "n"):
        return 0.0
    # Explicitly reject the textual non-finite literals Python's float() accepts
    # ("nan", "inf", "-inf", "infinity"): a connector must never inject a
    # non-finite feature value from a CSV/XML source.
    if low.lstrip("+-") in ("nan", "inf", "infinity"):
        return default
    try:
        return _finite_or_default(float(s), default)
    except ValueError:
        return default


def _apply_transform(col: np.ndarray, transform: str | None) -> np.ndarray:
    if transform is None or transform == "raw":
        return col
    if transform == "zscore":
        mu = float(col.mean())
        sd = float(col.std())
        return (col - mu) / sd if sd > 1e-9 else col - mu
    if transform == "log1p":
        return np.log1p(np.clip(col, 0, None))
    if transform == "bool":
        return (col != 0).astype(np.float64)
    raise ValueError(f"unknown transform {transform!r}")


# ---------------------------------------------------------------------------
# Source readers — each returns a list of flat dict records keyed by source_key.
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))


# Strip XML namespaces so the feature map can use plain dotted element paths
# (e.g. "Document/FIToFICstmrCdtTrf/CdtTrfTxInf/IntrBkSttlmAmt") regardless of
# the ISO 20022 namespace URN.
def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _iso_find(elem: ET.Element, dotted_path: str) -> str | None:
    """Resolve a namespace-agnostic dotted path against an element subtree.

    Matches by local element name at each step. Returns the text of the first
    match (or its first attribute if the leaf has no text but carries e.g. Ccy).
    """
    parts = dotted_path.split("/")
    # Allow the path to start at or below the document root.
    nodes = [elem]
    start = 0
    if parts and _localname(elem.tag) == parts[0]:
        start = 1
    for part in parts[start:]:
        nxt: list[ET.Element] = []
        for n in nodes:
            nxt.extend(c for c in n.iter() if _localname(c.tag) == part)
        if not nxt:
            return None
        nodes = nxt
    node = nodes[0]
    if node.text and node.text.strip():
        return node.text.strip()
    # leaf value sometimes lives in an attribute (rare); return first attr value
    if node.attrib:
        return next(iter(node.attrib.values()))
    return None


def _read_iso20022(path: str, fmap: FeatureMap) -> list[dict[str, Any]]:
    """Parse a pacs.008 / pain.001 message into one record per transaction.

    A single ISO 20022 credit-transfer message can carry many transaction-info
    blocks (CdtTrfTxInf). We emit one record per such block, resolving each
    declared iso_path *within that block* (falling back to the whole document
    for header-level fields like debtor info).
    """
    tree = ET.parse(path)
    root = tree.getroot()
    # Transaction blocks: pacs.008 -> CdtTrfTxInf; pain.001 -> CdtTrfTxInf too.
    tx_blocks = [e for e in root.iter() if _localname(e.tag) == "CdtTrfTxInf"]
    if not tx_blocks:
        tx_blocks = [root]

    records: list[dict[str, Any]] = []
    for block in tx_blocks:
        rec: dict[str, Any] = {}
        for name, spec in fmap.features.items():
            if not spec.iso_path:
                continue
            val = _iso_find(block, spec.iso_path)
            if val is None:
                val = _iso_find(root, spec.iso_path)  # header-level fallback
            rec[name] = val
        records.append(rec)
    return records


class ConnectorRuntime:
    """Resolves a FeatureMap + its source into (X, y) numpy arrays."""

    def __init__(self, fmap: FeatureMap):
        self.fmap = fmap

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        if self.fmap.source_kind == "csv":
            records = _read_csv(self.fmap.source_path)
            return self._project_csv(records)
        if self.fmap.source_kind == "iso20022":
            records = _read_iso20022(self.fmap.source_path, self.fmap)
            return self._project_iso(records)
        raise ValueError(f"unsupported source kind {self.fmap.source_kind!r}")

    # ---- projection ------------------------------------------------------

    def _empty_matrix(self, n: int) -> np.ndarray:
        X = np.zeros((n, FEATURE_DIM), dtype=np.float64)
        # apply per-feature defaults
        for col_idx, fname in enumerate(FEATURE_ORDER):
            spec = self.fmap.spec_for(fname)
            if spec is not None and spec.default:
                X[:, col_idx] = spec.default
        return X

    def _fill_column(self, X: np.ndarray, col_idx: int, spec: FeatureSpec,
                     raw_values: list[Any]) -> None:
        col = np.array([_to_float(v, spec.default) for v in raw_values], dtype=np.float64)
        col = _apply_transform(col, spec.transform)
        X[:, col_idx] = col

    def _project_csv(self, records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        n = len(records)
        X = self._empty_matrix(n)
        for col_idx, fname in enumerate(FEATURE_ORDER):
            spec = self.fmap.spec_for(fname)
            if spec is None or spec.source_key is None:
                continue
            raw_values = [r.get(spec.source_key) for r in records]
            self._fill_column(X, col_idx, spec, raw_values)

        label_col = self.fmap.label_column
        if label_col:
            y = np.array(
                [int(round(_to_float((r.get(label_col)), 0.0))) for r in records],
                dtype=np.int64,
            )
        else:
            y = np.zeros(n, dtype=np.int64)
        return X, y

    def _project_iso(self, records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        n = len(records)
        X = self._empty_matrix(n)
        for col_idx, fname in enumerate(FEATURE_ORDER):
            spec = self.fmap.spec_for(fname)
            if spec is None or spec.iso_path is None:
                continue
            raw_values = [r.get(fname) for r in records]
            self._fill_column(X, col_idx, spec, raw_values)
        # payment messages carry no label — SE supplies a constant for training
        y = np.full(n, int(self.fmap.iso_label), dtype=np.int64)
        return X, y


def run_connector(feature_map_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: load a feature map from disk and produce (X, y)."""
    fmap = load_feature_map(feature_map_path)
    return ConnectorRuntime(fmap).load()
