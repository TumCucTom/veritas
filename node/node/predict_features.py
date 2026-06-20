"""Map an inbound /predict transaction dict onto a FEATURE_DIM vector.

Mirrors the "safe-account mule" signature the engine trains against
(benign on generic fraud axes, distinctive on the campaign feature) so the
served model and the trained model speak the same feature contract. Callers
may pass any subset of the named features; absent ones default to 0.
"""
from __future__ import annotations

import numpy as np

from veritas_core.data import FEATURE_DIM

from .connectors.feature_map import FEATURE_ORDER


def transaction_to_features(txn: dict) -> np.ndarray:
    x = np.zeros(FEATURE_DIM, dtype=np.float64)
    # Direct named pass-through for any feature in the contract.
    for idx, name in enumerate(FEATURE_ORDER):
        if name in txn:
            try:
                x[idx] = float(txn[name])
            except (TypeError, ValueError):
                pass
    # If the caller only signalled a campaign match, synthesise the mule shape
    # (benign generic axes, strong campaign signature) so it scores as the
    # typology the federated model learned.
    if "campaignSignature" in txn or "campaignSig" in txn:
        sig = float(txn.get("campaignSignature", txn.get("campaignSig", 1.0)))
        x[FEATURE_ORDER.index("campaignSig")] = 1.5 * sig
        x[FEATURE_ORDER.index("amount")] = -0.5
        x[FEATURE_ORDER.index("velocity")] = -0.5
        x[FEATURE_ORDER.index("fanout")] = -0.5
        x[FEATURE_ORDER.index("accountAge")] = -2.0
    return x
