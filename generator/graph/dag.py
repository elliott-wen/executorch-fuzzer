"""dag.py — the generated computation graph.

A functional DAG. Each node is an op plus a full argument list in parameter order, where
every argument is one of:
    ('leaf',  i)    a graph input, LEAVES[i]
    ('ref',   nid)  the output of an earlier node
    ('const', v)    a baked value

This is the structure only. Turning it into a runnable script lives in render.py, because
how a graph is WRITTEN OUT is a serialization format that changes for entirely different
reasons than how a graph is built.
"""

from __future__ import annotations

from typing import Any

import torch


class Graph:
    """The DAG under construction, and the source it renders to."""

    def __init__(self) -> None:
        self.leaves: list[torch.Tensor] = []
        self.nodes: list[tuple[Any, list]] = []

    def add(self, op, slots: list) -> int:
        """Append a node and return its id.

        Callers pass ('leaf', tensor) | ('baked_leaf', tensor) | ('ref', nid) |
        ('const', v). A plain leaf becomes a graph INPUT, regenerated from shape and dtype
        when the script runs. A baked leaf instead freezes the solved values into the
        source as a constant — which is required for a port whose DATA is constrained
        (an index, a target), since re-rolling it through `_make` would produce
        out-of-range garbage the op rejects.
        """
        resolved = []
        for kind, value in slots:
            if kind == "leaf":
                self.leaves.append(value)
                resolved.append(("leaf", len(self.leaves) - 1))
            elif kind == "baked_leaf":
                resolved.append(("const", value))
            else:
                resolved.append((kind, value))
        self.nodes.append((op, resolved))
        return len(self.nodes) - 1

    def sink_ids(self) -> list[int]:
        """Nodes nobody reads — the graph's outputs."""
        consumed = {v for _, slots in self.nodes for kind, v in slots if kind == "ref"}
        return [nid for nid in range(len(self.nodes)) if nid not in consumed]

    def describe(self) -> str:
        """One-line op chain, e.g. `n0=add(L0,L1) ; n1*=relu(n0)` (* marks an output).
        This is what names a graph in logs and failure reports."""
        sinks = set(self.sink_ids())
        parts = []
        for nid, (op, slots) in enumerate(self.nodes):
            args = ",".join("L%d" % v if k == "leaf" else "n%d" % v if k == "ref" else "·"
                            for k, v in slots)
            parts.append(f"n{nid}{'*' if nid in sinks else ''}={op.label}({args})")
        return " ; ".join(parts)

    def emit(self, seed: int = 0, nonfinite_prob=None) -> str | None:
        """Render as a standalone script — see render.render()."""
        from mobile.generator.graph.render import render
        return render(self, seed=seed, nonfinite_prob=nonfinite_prob)
