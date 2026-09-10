"""loader.py — load one op's Z3 precondition out of pytorch_constraints/.

Each `pytorch_constraints/<mangled symbol>/slice_*.py` is GENERATED Python: it
rebuilds the op's argument variables and returns the Z3 constraints describing the
paths on which that op RAISES (a reachable TORCH_CHECK). Callers assert the negation
to stay on valid inputs.

Loading is lazy and per-symbol, because a run only ever uses a fraction of what is on
disk: 1346 symbol folders, 47.5 MB of generated source, against a few hundred ops a run
actually wants. `available()` answers "what is there?" from directory entries alone —
exec'ing nothing — so a caller narrows first and pays only for what it keeps. Which
symbols are worth loading is decided by graph/catalog/ops.tsv, not here. (The loader this replaces exec'd all 1346 up front and filtered
after, which is where most of the ~75s worker startup went.) Deciding which ops matter
— blocklists, backend allowlists, overload choice — is deliberately not this module's
business; its whole contract is "given a symbol, give me its constraints".

A slice is an ordinary module doing an ordinary import: it pulls the vocabulary from
`mobile.generator.constraints.model` by name, so exec'ing one needs no `sys.path` or
`sys.modules` staging. (Upstream generates them wired to an absolute path in its own
workspace — rewritten once, on vendoring, to import ours instead.)
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mobile.generator.constraints.model import _ARG_BUILDERS

_ROOT = Path(__file__).resolve().parent / "pytorch_constraints"


@dataclass(frozen=True)
class OpConstraints:
    """One op's loaded precondition. `bad` is a list-per-slice of Z3 expressions
    describing REACHABLE ERROR paths — a solver wants `Not(And(*slice))` for each."""

    symbol: str
    params: list[tuple[str, str]]   # [(arg name, model type)], from the first slice
    vars: dict[str, Any]            # {arg name: TensorVar/ScalarVar/IntArrayVar/...}
    bad: list[list]                 # per-slice error-path constraints
    axioms: list[list]              # per-slice axioms, for slices that carry them
    unresolved: frozenset[str]      # vars the extractor could not resolve


def root() -> Path:
    return _ROOT


def set_root(path) -> None:
    """Point the loader at a different constraints tree (fixtures in tests)."""
    global _ROOT
    _ROOT = Path(path)
    _CACHE.clear()


def available() -> list[str]:
    """Every symbol with a constraint folder. One `iterdir` — execs nothing."""
    if not _ROOT.is_dir():
        return []
    return sorted(p.name for p in _ROOT.iterdir()
                  if p.is_dir() and not p.name.startswith("__"))


def _exec_slice(path: Path):
    """Exec one slice file and hand back its module.

    Goes through importlib rather than `compile`+`exec` so the parse of these very
    large generated files is cached in __pycache__ and paid once per machine, not once
    per worker process. The module is never inserted into `sys.modules` — nothing
    imports a slice by name.
    """
    spec = importlib.util.spec_from_file_location(f"_slice_{path.parent.name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flatten(constraints: list) -> list:
    """Recursively flatten nested lists so Z3 never receives a list as an arg."""
    out = []
    for c in constraints:
        if isinstance(c, list):
            out.extend(_flatten(c))
        else:
            out.append(c)
    return out


_CACHE: dict[str, OpConstraints | None] = {}


def load(symbol: str) -> OpConstraints | None:
    """One op's constraints, or None if the folder holds no slices. Memoized.

    A malformed slice raises rather than being swallowed: a constraint that silently
    fails to load is an op that silently vanishes from the corpus, which is much harder
    to notice than a traceback.
    """
    if symbol not in _CACHE:
        _CACHE[symbol] = _load_uncached(symbol)
    return _CACHE[symbol]


def _load_uncached(symbol: str) -> OpConstraints | None:
    slices = sorted((_ROOT / symbol).glob("slice_*.py"))
    if not slices:
        return None
    params: list[tuple[str, str]] | None = None
    bad: list[list] = []
    axioms: list[list] = []
    unresolved: set[str] = set()
    for path in slices:
        mod = _exec_slice(path)
        if params is None:
            params = list(mod.get_params())
        unresolved |= set(mod.get_unresolved_vars())
        clist = _flatten(mod.get_constraints())
        if clist:
            bad.append(clist)
        if hasattr(mod, "get_axioms"):
            alist = _flatten(mod.get_axioms())
            if alist:
                axioms.append(alist)
    params = params or []
    # Variables are rebuilt from the param list rather than taken from the slice's own
    # get_vars(): Z3 consts are identified by name, so a fresh TensorVar('self') denotes
    # the same term the constraints were written against, and every consumer gets a
    # handle it can pin without reaching into the slice module.
    variables = {n: _ARG_BUILDERS[t](n) for n, t in params if t in _ARG_BUILDERS}
    return OpConstraints(symbol, params, variables, bad, axioms, frozenset(unresolved))
