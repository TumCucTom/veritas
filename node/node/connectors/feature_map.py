"""Declarative feature-map model + loader.

A feature_map.yaml looks like:

    version: 1
    source:
      kind: csv                       # or iso20022
      path: sample_data/transactions.csv
      label_column: is_fraud          # csv only; column holding 0/1 ground truth
    features:                         # ordered: must cover the FEATURE_DIM contract
      amount:        { from: amount,        transform: zscore }
      oldOrig:       { from: old_balance_orig }
      ...
      campaignSig:   { from: campaign_signature, default: 0.0 }
    # iso20022 sources use XPath-ish dotted ISO paths instead of `from`:
      amount:        { iso_path: "Document/.../IntrBkSttlmAmt", transform: zscore }

The runtime resolves each declared feature, in the declared order, into a
column of the (N x FEATURE_DIM) matrix. Features not declared default to 0.0.
The order of the resulting columns follows ``FEATURE_ORDER`` (the core contract),
not the YAML order, so the produced X always matches the model's expectation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from veritas_core.data import FEATURE_DIM

# The canonical column order of the FEATURE_DIM feature vector, from
# core/veritas_core/data.py:
#   amount, oldOrig, newOrig, oldDest, newDest, accountAge, velocity, fanout, isTransfer, campaignSig
FEATURE_ORDER = [
    "amount",
    "oldOrig",
    "newOrig",
    "oldDest",
    "newDest",
    "accountAge",
    "velocity",
    "fanout",
    "isTransfer",
    "campaignSig",
]
assert len(FEATURE_ORDER) == FEATURE_DIM, "FEATURE_ORDER must match FEATURE_DIM"

# Common snake_case / bank-native field names mapped to the canonical
# (camelCase) contract names. Real bank CSVs use names like ``account_age`` and
# ``velocity_1h``; this alias table lets such a feature_map.yaml load without the
# bank renaming its columns. Mirrors bank-connectors' FEATURE_ALIASES so the two
# connector implementations resolve names identically.
FEATURE_ALIASES = {
    "account_age": "accountAge",
    "velocity_1h": "velocity",
    "is_new_payee": "isTransfer",
    "is_transfer": "isTransfer",
    "campaign_signature": "campaignSig",
    "campaign_sig": "campaignSig",
    "old_orig": "oldOrig",
    "new_orig": "newOrig",
    "old_dest": "oldDest",
    "new_dest": "newDest",
}


@dataclass
class FeatureSpec:
    name: str
    source_key: str | None = None      # column name (csv) / friendly key
    iso_path: str | None = None        # dotted ISO 20022 element path (iso20022)
    transform: str | None = None       # None | "zscore" | "log1p" | "bool"
    default: float = 0.0


@dataclass
class FeatureMap:
    version: int
    source_kind: str                   # "csv" | "iso20022"
    source_path: str
    label_column: str | None
    features: dict[str, FeatureSpec] = field(default_factory=dict)
    # iso20022: a constant label applied to all rows parsed from the message
    # (payment messages don't carry fraud labels; SE supplies it for training).
    iso_label: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    def spec_for(self, name: str) -> FeatureSpec | None:
        return self.features.get(name)


def load_feature_map(path: str | Path) -> FeatureMap:
    p = Path(path)
    with p.open() as fh:
        doc = yaml.safe_load(fh) or {}

    source = doc.get("source", {}) or {}
    kind = source.get("kind")
    if kind not in ("csv", "iso20022"):
        raise ValueError(f"feature_map: unsupported source.kind {kind!r} (expected 'csv' or 'iso20022')")

    raw_source_path = source.get("path", "")
    # Resolve relative source paths against the feature_map's own directory.
    src_path = raw_source_path
    if raw_source_path and not Path(raw_source_path).is_absolute():
        src_path = str((p.parent / raw_source_path).resolve())

    features: dict[str, FeatureSpec] = {}
    for name, spec in (doc.get("features", {}) or {}).items():
        # Resolve snake_case / bank-native aliases to the canonical contract
        # name BEFORE validating, so a feature_map.yaml may declare e.g.
        # ``account_age`` and it is stored under ``accountAge``.
        canonical = FEATURE_ALIASES.get(name, name)
        if canonical not in FEATURE_ORDER:
            raise ValueError(
                f"feature_map: feature {name!r} is not part of the FEATURE_DIM contract {FEATURE_ORDER}"
            )
        spec = spec or {}
        features[canonical] = FeatureSpec(
            name=canonical,
            source_key=spec.get("from"),
            iso_path=spec.get("iso_path"),
            transform=spec.get("transform"),
            default=float(spec.get("default", 0.0)),
        )

    return FeatureMap(
        version=int(doc.get("version", 1)),
        source_kind=kind,
        source_path=src_path,
        label_column=source.get("label_column"),
        features=features,
        iso_label=int(source.get("label", 0)),
        raw=doc,
    )
