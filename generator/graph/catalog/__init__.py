"""Which operators exist for us, which torch overload each one is, and what
ExecuTorch can do with it.

  ops         the resolved table: symbol -> (op_name, tier, note). Resolved and verified
              offline; its docstring records how, since the tooling is not kept here.
  table       lookup / symbols / counts over it
  blocklist   ops we never want to generate, with the reason recorded (curated)
  lookup      op name -> torch callable
  load        load_ops — the filtered set, constraints loaded lazily
"""

from __future__ import annotations

from .blocklist import is_blocked
from .load import load_ops, tiers_for
from .lookup import callable_for
from .table import LOWERABLE_TIERS, counts, lookup, op_name_for, symbols, tier_for

__all__ = ["load_ops", "tiers_for", "is_blocked", "callable_for",
           "lookup", "op_name_for", "tier_for", "symbols", "counts", "LOWERABLE_TIERS"]
