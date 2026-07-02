"""opnode.py — the per-op constraint unit.

An OpNode wraps one approved_constraints/<symbol>/ folder. It can:
  - load that op's Z3 precondition (across all slices) and parameter list,
  - generate one valid concrete input set (optionally with some Tensor ports
    pinned to a producer's output spec),
  - report which params are Tensor consumer ports for graph wiring.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import torch  # noqa: E402
from z3 import And, Not, Select, IntVal, is_const, Solver  # noqa: E402

from mobile.gen import concretize as _concretize  # noqa: E402
from mobile.gen.z3engine import model as _z3model  # noqa: E402
from mobile.gen.z3engine.model import (  # noqa: E402
    _ARG_BUILDERS, TensorVar, TensorListVar, OptVar, IntArrayVar,
    MAX_DIM, MAX_INT_ARRAY_LEN,
)
from mobile.gen.z3engine.op_lookup import find_op_for_symbol  # noqa: E402

# The constraint slices exec'd in OpNode._load do a bare `from model import ...`. Expose the
# real model module under that bare name so they resolve — explicit, vs. a global sys.path hack.
sys.modules.setdefault("model", _z3model)
from mobile.gen.z3engine.sampler import _diverse_model  # noqa: E402

from mobile.gen.ops.blocklist import is_blocked  # noqa: E402
from mobile.gen.ops.allowlist import (  # noqa: E402
    is_executorch_op,
)

# opnode.py lives at mobile/gen/ops/; approved_constraints/ is vendored INSIDE the mobile
# package (mobile/approved_constraints/) so the package references only its own content.
APPROVED = Path(__file__).resolve().parents[2] / "approved_constraints"

# Tractability bounds for generated tensors (kept small on purpose).
_MAX_SIZE = 4

# torch.dtype → model ScalarType int code (inverse of the concretizer's map).
# Used to pin a producer output's dtype into a downstream op's solver.
_DT2CODE = {v: k for k, v in _concretize._torch_dtypes().items()}

# Concretizer hook — defaults to z3/concretize.build_call_args. A consumer can swap
# in an alternative (e.g. the mobile pipeline injects its own tweakable copy via
# set_build_call_args) WITHOUT forking this module. Called at request time below.
_build_call_args = _concretize.build_call_args


def set_build_call_args(fn) -> None:
    """Override the concretizer used by OpNode.generate (process-local)."""
    global _build_call_args
    _build_call_args = fn


def _const_names(exprs) -> set[str]:
    """Collect every constant/variable decl name appearing in the constraint
    exprs (walking the Z3 DAG once)."""
    names: set[str] = set()
    seen: set[int] = set()

    def walk(e):
        eid = e.get_id()
        if eid in seen:
            return
        seen.add(eid)
        if is_const(e):
            names.add(e.decl().name())
        for ch in e.children():
            walk(ch)

    for e in exprs:
        if hasattr(e, "children"):
            walk(e)
    return names


def _eval_real(model, expr):
    """Evaluate a Z3 Real to a float (or None if non-numeric)."""
    try:
        v = model.eval(expr, model_completion=True)
        if hasattr(v, "as_long"):
            return float(v.as_long())
        if hasattr(v, "numerator_as_long"):
            return v.numerator_as_long() / v.denominator_as_long()
        return float(str(v).replace("?", ""))
    except Exception:
        return None


# ── Z3 solver assembly (inlined from the former z3engine/solver.py; OpNode is the
#    only consumer) ───────────────────────────────────────────────────────────
Z3_TIMEOUT_MS = 10_000   # Z3 wall-clock budget per solver.check() call (ms)


def _make_solver() -> Solver:
    s = Solver()
    s.set("timeout", Z3_TIMEOUT_MS)
    return s


def _flatten(constraints: list) -> list:
    """Recursively flatten nested lists so Z3 never receives a list as an arg."""
    out = []
    for c in constraints:
        if isinstance(c, list):
            out.extend(_flatten(c))
        else:
            out.append(c)
    return out


def _add_domain_axioms(solver: Solver, named_vars: dict) -> None:
    """Add per-type bounds so the solver can produce sane witnesses."""
    for var in named_vars.values():
        if hasattr(var, "axioms"):
            solver.add(*var.axioms())
    # Pairwise dtype compatibility: prevent Z3 from pairing uint16/32/64 or Float8
    # types across tensors. promote_types throws for those cross-type combos, so any
    # generated input would fail at runtime for the wrong reason.
    from mobile.gen.z3engine.model import compatible_dtypes
    dtypes = [var.dtype for var in named_vars.values() if isinstance(var, TensorVar)]
    for i in range(len(dtypes)):
        for j in range(i + 1, len(dtypes)):
            solver.add(compatible_dtypes(dtypes[i], dtypes[j]))


class OpNode:
    """Loads one op's approved constraint; generates valid inputs, optionally with
    some Tensor ports pinned to a producer's output spec."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        op, op_name, _dk = find_op_for_symbol(symbol)
        self.op = op            # resolved aten callable
        self.op_name = op_name
        self.label = op_name or symbol[:24]
        self._load()

    def _load(self) -> None:
        subfolder = APPROVED / self.symbol
        named_params = None
        named_vars: dict[str, Any] = {}
        bad: list[list] = []
        axioms: list[list] = []
        for py in sorted(subfolder.glob("slice_*.py")):
            spec = importlib.util.spec_from_file_location("_slice", py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if named_params is None:
                named_params = mod.get_params()
            if not named_vars:
                named_vars = {
                    n: _ARG_BUILDERS[mt](n)
                    for n, mt in named_params
                    if mt in _ARG_BUILDERS
                }
            clist = _flatten(mod.get_constraints())
            if hasattr(mod, "get_axioms"):
                a = _flatten(mod.get_axioms())
                if a:
                    axioms.append(a)
            if clist:
                bad.append(clist)
        self.named_params = named_params or []
        self.named_vars = named_vars
        self.bad = bad
        self.axioms = axioms
        # Tensor consumer ports = positional Tensor params, excluding any `out`.
        self.tensor_ports = [
            n for n, t in self.named_params if t == "Tensor" and n != "out"
        ]
        # Ports carrying a VALUE precondition (the constraint bounds their data
        # via `<port>.data_lo`/`.data_hi` — e.g. index/target/probability ports).
        # A fresh-leaf input gets clamped into range by the concretizer, but a
        # producer wired into such a port does NOT — so graph_build.py inserts a range
        # adapter (clamp) using the solved [lo, hi] from generate().
        names = _const_names(c for cl in bad for c in cl)
        self.range_ports = {
            p for p in self.tensor_ports
            if f"{p}.data_lo" in names or f"{p}.data_hi" in names
        }

    # -- solver assembly --------------------------------------------------------

    def _add_size_axioms(self, solver) -> None:
        def _bound_tv(tv):
            solver.add(tv.ndim <= MAX_DIM)
            for i in range(MAX_DIM):
                cell = Select(tv.sizes, IntVal(i))
                solver.add(cell >= 0, cell <= _MAX_SIZE)

        for var in self.named_vars.values():
            if isinstance(var, TensorVar):
                _bound_tv(var)
            elif isinstance(var, TensorListVar):
                for tv in var.tensors:
                    _bound_tv(tv)
            elif isinstance(var, OptVar) and isinstance(var.value, TensorVar):
                _bound_tv(var.value)
            elif isinstance(var, IntArrayVar):
                # CRITICAL: bound int[] elements — an unbounded `size`/`shape`
                # arg (factory ops, reshape, empty_permuted) would otherwise let
                # the concretizer allocate a giant tensor and OOM/abort the worker.
                for i in range(MAX_INT_ARRAY_LEN):
                    cell = Select(var.data, IntVal(i))
                    solver.add(cell >= -MAX_DIM, cell <= _MAX_SIZE)

    def _solver(self, pins=None):
        s = _make_solver()
        _add_domain_axioms(s, self.named_vars)
        self._add_size_axioms(s)
        for a in self.axioms:
            s.add(*a)
        for c in self.bad:
            s.add(Not(And(*c)))
        if pins:
            s.add(*pins)
        return s

    def pin_for(self, port_name: str, t: torch.Tensor):
        """Z3 constraints forcing port `port_name` to match tensor t's spec."""
        tv = self.named_vars[port_name]
        pins = [tv.ndim == t.dim()]
        for i in range(t.dim()):
            pins.append(Select(tv.sizes, IntVal(i)) == t.size(i))
        code = _DT2CODE.get(t.dtype)
        if code is not None:
            pins.append(tv.dtype == code)
        return pins

    def generate(self, rng, pins=None):
        """Return ConcreteArgs for one valid input, or None if UNSAT/exhausted.
        Attaches `.ranges`: {range_port: (lo, hi)} — the solved integer value
        bounds, matching the clamp the concretizer applied to that fresh leaf."""
        s = self._solver(pins)
        m = _diverse_model(s, self.named_vars, rng)
        if m is None:
            return None
        c = _build_call_args(
            self.named_params, self.named_vars, m,
            rng=rng, data_mode="random", op_name=self.op_name,
        )
        ranges = {}
        for p in self.range_ports:
            tv = self.named_vars.get(p)
            if tv is None:
                continue
            lo = _eval_real(m, tv.data_lo)
            hi = _eval_real(m, tv.data_hi)
            if lo is None or hi is None:
                continue
            ilo, ihi = math.ceil(lo), math.floor(hi)
            if ihi >= ilo:  # skip empty/degenerate ranges (let those stay leaves)
                ranges[p] = (ilo, ihi)
        c.ranges = ranges
        return c


def load_opnodes(executorch: bool = False) -> list["OpNode"]:
    """Load all approved ops minus the blocklist (that resolve and have params).
    This is the op-selection step of generation — owned here, not in the driver.

    When `executorch` is set, ops are additionally restricted to those ExecuTorch
    can run (see executorch_allowlist.EXECUTORCH_OPS); the blocklist still applies
    on top, so the result is the ExecuTorch set minus blocklisted noise ops."""
    symbols = sorted(p.name for p in APPROVED.iterdir() if p.is_dir())
    print(f"Loading constraints for {len(symbols)} approved ops "
          f"(minus blocklist{', ExecuTorch-only' if executorch else ''}) ...",
          flush=True)
    opnodes, n_load_err, n_blocked, n_non_et = [], 0, 0, 0
    for sym in symbols:
        try:
            node = OpNode(sym)
        except Exception:
            n_load_err += 1
            continue
        if node.op is None or not node.named_params:
            continue
        if is_blocked(node.op_name):
            n_blocked += 1
            continue
        if executorch and not is_executorch_op(node.op_name):
            n_non_et += 1
            continue
        opnodes.append(node)
    et_note = f", {n_non_et} non-ExecuTorch" if executorch else ""
    print(f"  Usable: {len(opnodes)} ops ({n_blocked} blocklisted{et_note}, "
          f"{n_load_err} load errors)", flush=True)
    return opnodes
