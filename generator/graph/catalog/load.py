"""load.py — build the set of operators this generator may emit.

Selection runs on the table and the op NAME first, and only then loads constraints. That
ordering is the whole point: there are 1346 constraint folders, 47.5 MB of generated
Python that has to be exec'd and turned into Z3 terms, and a run wants a few hundred of
them. Deciding first costs a dict lookup per symbol; deciding after costs the rest.

Which overload a symbol is, and whether ExecuTorch can take it, are both answered by
ops.tsv — resolved and verified offline (see table.py). Nothing here searches candidates
or repairs a wrong guess.
"""

from __future__ import annotations

from mobile.generator.constraints import load
from mobile.generator.graph.catalog import table as _table
from mobile.generator.graph.catalog.blocklist import is_blocked
from mobile.generator.graph.catalog.lookup import callable_for
from mobile.generator.graph.op import Op


def _fixed_array_lengths(fn) -> dict[str, int]:
    """{arg name: N} for each fixed-arity `int[N]` argument, read off the real JIT schema.

    The model collapses `int[N]` to a plain `int[]` and loses N, so the solver would leave
    the length free and rarely land on the one value the op accepts. Argument.N carries it
    (None means dynamic).
    """
    schema = getattr(fn, "_schema", None)
    if schema is None:
        return {}
    return {a.name: int(a.N) for a in schema.arguments if getattr(a, "N", None) is not None}


def tiers_for(composable: bool = False) -> tuple[str, ...]:
    """Which table tiers a run wants.

    `core` alone by default: those lower AND run, so they are the only ops that can yield a
    differential comparison. `composable` adds the 759 ops that lower only because to_edge
    decomposes them — the sole way to exercise the decomposition pass, but about half of
    them then hit OperatorMissing at run time, testing export without producing a diff.
    """
    return ("core", "composable") if composable else ("core",)


def load_ops(accept=None, verbose: bool = True, composable: bool = False) -> list[Op]:
    """Every operator the run may emit, with its constraints loaded.

    Loading the core tier costs ~3.2 s (exec'ing a symbol's generated Z3 is ~14.5 ms), paid
    once per worker process. Deferring it per-op was tried and removed: a worker handles many
    indices and draws nearly the whole catalog within its first ~50 graphs, so laziness only
    moved when the cost was paid while making Op harder to read.

    `accept` takes an op NAME and narrows the set further (one op, one family) without this
    module knowing why; the blocklist always applies on top.
    """
    symbols = _table.symbols(tiers_for(composable))
    ops: list[Op] = []
    skipped_policy = skipped_unresolved = skipped_empty = 0

    for symbol in symbols:
        op_name = _table.op_name_for(symbol)
        if op_name is None or is_blocked(op_name) or (accept is not None and not accept(op_name)):
            skipped_policy += 1
            continue
        fn = callable_for(op_name)
        if fn is None:                          # the table names an op this build lacks
            skipped_unresolved += 1
            continue
        constraints = load(symbol)              # the expensive step, paid only here
        if constraints is None or not constraints.params:
            skipped_empty += 1
            continue
        ops.append(Op(symbol=symbol, op_name=op_name, fn=fn, constraints=constraints,
                      fixed_array_len=_fixed_array_lengths(fn)))

    if verbose:
        print(f"ops: {len(ops)} of {len(symbols)} lowerable symbols "
              f"({skipped_policy} filtered, {skipped_unresolved} unresolved, "
              f"{skipped_empty} without constraints)", flush=True)
    return ops
