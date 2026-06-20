"""Thin re-export of the CANONICAL live-ensemble module.

The single source of truth for the live-ensemble wire layout, per-block DP
budgets and the federation helpers is ``veritas_core.live_ensemble``. This
module used to carry a DUPLICATE implementation; it now simply re-exports the
core symbols so existing ``node`` imports (``LIVE_ENSEMBLE_DIM``,
``noise_sensitivity``, ``genesis_weights``, ``privatize``, ``block_slice``,
``BLOCK_BUDGETS``) keep working while there is exactly ONE implementation.

New code should prefer ``from veritas_core import live_ensemble`` directly.
"""
from __future__ import annotations

from veritas_core.live_ensemble import (  # noqa: F401
    BLOCK_BUDGETS,
    LIVE_ENSEMBLE_DIM,
    block_slice,
    block_slices,
    clip_blocks,
    genesis_weights,
    init_weights,
    noise_sensitivity,
    predict_proba,
    privatize,
    recall,
    train_local,
    weight_dim,
)

# Back-compat alias for the per-block clip budgets under the node's historical
# name (``config.PER_BLOCK_NORM`` etc.). Points at the canonical budgets.
PER_BLOCK_NORM = BLOCK_BUDGETS

__all__ = [
    "BLOCK_BUDGETS",
    "LIVE_ENSEMBLE_DIM",
    "PER_BLOCK_NORM",
    "block_slice",
    "block_slices",
    "clip_blocks",
    "genesis_weights",
    "init_weights",
    "noise_sensitivity",
    "predict_proba",
    "privatize",
    "recall",
    "train_local",
    "weight_dim",
]
