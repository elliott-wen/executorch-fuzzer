"""table.py — queries over the resolved operator table.

`table_data.py` holds the data: for every symbol with constraints, which torch overload it is and
what ExecuTorch can do with it. This module is just the handful of questions the generator
asks of it. See table_data.py for how the table was produced and why it is a snapshot rather than
a specification.

Tiers:
    core        lowers to a .pte, and the runtime registers this op's own kernel
    composable  lowers only because to_edge decomposes it into core ops
    n/a         unusable — no .pte came out, or no valid call could be built at all.
                Kept in the table rather than deleted so a rebuild can tell "checked and
                rejected" from "never checked"; the entry's note says which it was.
"""

from __future__ import annotations

from mobile.generator.ops.table_data import OPS

#: tiers whose ops actually produce a .pte, in widening order
LOWERABLE_TIERS = ("core", "composable")


def lookup(symbol: str) -> tuple[str, str, str] | None:
    """(op_name, tier, note) for a symbol, or None if it isn't in the table."""
    return OPS.get(symbol)


def op_name_for(symbol: str) -> str | None:
    entry = OPS.get(symbol)
    return entry[0] if entry and entry[0] else None


def tier_for(symbol: str) -> str | None:
    entry = OPS.get(symbol)
    return entry[1] if entry else None


def symbols(tiers=LOWERABLE_TIERS) -> list[str]:
    """Every symbol whose tier is one of `tiers`, in table order."""
    wanted = frozenset(tiers)
    return [s for s, entry in OPS.items() if entry[1] in wanted]


def counts() -> dict[str, int]:
    """{tier: n} — a one-line summary of what the table says."""
    out: dict[str, int] = {}
    for entry in OPS.values():
        out[entry[1]] = out.get(entry[1], 0) + 1
    return out
