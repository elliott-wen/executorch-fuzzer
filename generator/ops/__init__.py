"""ops — the operators we can emit, each ready to produce a valid call.

An entry is more than a name — by the time `load_ops` hands one over it carries the torch
callable, the op's precondition as Z3, and the ability to produce concrete arguments that
satisfy it. That is the whole contract this package offers the graph builder: `Op`s that
can generate themselves. How they are then wired into a DAG is `graph`'s business, and
nothing here knows about it.

  table_data  the resolved table: symbol -> (op_name, tier, note). Produced offline; its
              docstring records how, since the tooling is not kept here.
  table       lookup / symbols / counts over it
  blocklist   ops we never generate, with the reason recorded — general, not per-target:
              nondeterminism and shapes the generator cannot produce, on any runtime
  lookup      op name -> torch callable
  op          Op — one operator; generate() -> concrete valid arguments
  solver      assemble an op's Z3 precondition; draw diverse samples from it
  load        load_ops — the selected set, with constraints loaded

    from mobile.generator.ops import load_ops

    ops = load_ops(target="portable")     # what this runtime can run, and what it demands
"""

from __future__ import annotations

from .blocklist import is_blocked
from .load import load_ops, tiers_for
from .lookup import callable_for
from .op import Op
from .table import LOWERABLE_TIERS, counts, lookup, op_name_for, symbols, tier_for

__all__ = ["load_ops", "tiers_for", "is_blocked", "callable_for", "Op",
           "lookup", "op_name_for", "tier_for", "symbols", "counts", "LOWERABLE_TIERS"]
