"""Graph synthesis: pick operators, solve them, and assemble a runnable DAG.

The operator catalog and the graph builder live together because they are one loop, not
two layers — growing a graph means asking an op to solve itself against a shape the graph
already produced, and the answer decides whether the node can be attached at all.

  catalog/    the resolved op table (ops.tsv): which overload, and which tier
  solver      assemble an op's Z3 precondition; draw diverse samples from it
  op          Op — one operator; generate() → concrete valid arguments
  probe       validate a candidate node on meta tensors, and learn its output shape
  meta_impls  meta rules for ops the runtime has none for
  adapter     glue nodes that connect a producer into a mismatched port
  build       build_graph — the growth loop
  dag         Graph — the DAG structure
  render      a Graph, written out as a standalone script

    ops = load_ops()
    graph = build_graph(rng, ops, n_nodes=8, seed_op=ops[i % len(ops)])
    source = graph.emit(seed=i)     # defines g(*LEAVES); run it anywhere
"""

from __future__ import annotations

from .build import build_graph
from .catalog import is_blocked, load_ops, tiers_for
from .dag import Graph
from .op import Op

__all__ = [
    "Graph", "Op", "build_graph", "load_ops",
    "tiers_for", "is_blocked",
]
