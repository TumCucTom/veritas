"""Veritas Tier 1 Bank Node.

A single bank's deployable node: holds its own local data, trains locally,
federates DP-protected model deltas with the control plane over HTTP, and
serves real-time predictions. Splits the in-process N-bank ``Engine`` in
``core/veritas_core/engine.py`` into ONE bank + a networked federation client.
"""

__version__ = "0.1.0"
