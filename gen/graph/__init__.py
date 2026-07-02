"""Graph assembly: build a random op graph and emit its source.

- ir:      GenGraph — the in-memory graph IR + source emitter
- emit:    op-name qualname / constant-repr helpers used by the emitter
- adapter: shape/dtype adapter nodes that connect mismatched ports
- build:   build_graph — grow a random graph from the op catalog
"""
from __future__ import annotations

from .ir import GenGraph
from .build import build_graph

__all__ = ["GenGraph", "build_graph"]
