"""Graph synthesis: given a set of operators, grow a DAG and write it out.

The operators arrive ready to use — `build_graph` takes them as an argument and only ever
asks one to `generate()` a call. So nothing here loads constraints, builds a solver or
knows what a target is; this package imports torch and itself, and that is all. Which
operators exist, and what a runtime demands of them, is `ops`' business.

Growth is incremental rather than a whole-graph solve: a node is solved ALONE, pinned
against the concrete tensor its producer already returned. That is why a producer's output
is a real tensor by the time anything is wired onto it, and why no two nodes ever share a
Z3 problem.

  build       build_graph — the growth loop
  dag         Graph — the DAG structure
  probe       validate a candidate node on meta tensors, and learn its output shape
  meta_impls  meta rules for ops the runtime has none for
  adapter     glue nodes that connect a producer into a mismatched port
  render      a Graph, written out as a standalone script

    from mobile.generator.ops import load_ops
    from mobile.generator.graph import build_graph

    ops = load_ops(target="portable")
    graph = build_graph(rng, ops, n_nodes=8, seed_op=ops[i % len(ops)])
    source = graph.emit(seed=i)     # defines g(*LEAVES); run it anywhere
"""

from __future__ import annotations

from .build import build_graph
from .dag import Graph

__all__ = ["Graph", "build_graph"]
