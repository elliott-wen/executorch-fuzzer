"""
test_valid_inputs.py — Test an ATen operator's pass rate on Z3-generated valid inputs.

Given a constraints/<sym>/ subfolder, this tool:
  1. Loads all usable slice constraint files (arg_map comes from each slice's
     own get_params(); no slices_ir JSON needed).
  2. Builds a Z3 solver encoding NOT(any bad condition) across all slices.
  3. Enumerates up to --samples diverse valid inputs via Z3 model enumeration.
  4. Runs the ATen operator on each input in a forked child.
  5. Reports pass rate: fraction of non-skip calls that complete without RuntimeError.

Inputs are always generated diversely: random shape/dtype/scalar variation driven
through the solver, with random tensor data.  Most samples are non-degenerate (so
ops execute their real kernel) but ~10% are deliberate edge cases — empty tensors,
0-dim scalar tensors, or inf/nan values — to keep those code paths covered.

Usage:
    cd /data/jwen929/pytorch/z3
    python3 test_valid_inputs.py <constraints_subfolder> \\
        [--samples 1000] [--pass-rate 0.99] [--verbose]

Exit codes:
    0 = PASS  (pass rate >= --pass-rate threshold)
    1 = FAIL  (pass rate below threshold)
    2 = ERROR (no op found, no usable slices, etc.)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import torch  # import once; forked children inherit the loaded module
import random

from z3 import And, Not, Or, Implies, IntVal, RealVal, Select, BoolRef, ArithRef, sat

from mobile.gen.concretize import build_call_args
from mobile.engine.model import (
    TensorVar, IntArrayVar, TensorListVar, ScalarListVar, OptVar, ScalarVar,
    _ARG_BUILDERS, MAX_DIM,
    MAX_INT_ARRAY_LEN,
    INT32, INT64, FLOAT16, FLOAT32, FLOAT64, BFLOAT16, BOOL, COMPLEX64, COMPLEX128,
)
from mobile.engine.op_lookup import find_op_for_symbol
import importlib.util

from mobile.engine.verifier import (
    _add_domain_axioms,
    _call_op_forked,
    _flatten,
    _make_solver,
)


_MAX_NDIM = MAX_DIM   # keep in sync with model.MAX_DIM (5D tensors for *3d ops)
_MAX_SIZE = 4


def _split_by_schema(op, args: list) -> tuple[list, dict]:
    """Split a flat arg list into (positional_args, kwarg_dict) using the op's JIT schema.

    ATen ops declare kwarg-only args (after `*` in the schema, e.g. `out=`) that
    must be passed as keyword arguments.  Passing them positionally raises TypeError.
    """
    try:
        import torch
        schema = op._schema
        pos, kw = [], {}
        first_tensor_dtype = None
        for val, sa in zip(args, schema.arguments):
            if sa.kwarg_only:
                # Required kwarg-only args (no default, e.g. div.out_mode's
                # `rounding_mode`) must be passed even when None, or the call
                # raises "missing value for argument".
                required = not sa.has_default_value() if hasattr(sa, "has_default_value") else False
                if val is not None or required:
                    kw[sa.name] = val
            else:
                pos.append(val)
                if first_tensor_dtype is None and isinstance(val, torch.Tensor):
                    first_tensor_dtype = val.dtype
        # Synthesize missing kwarg-only Tensor outputs.  For meta-only structured
        # ops (e.g. norm.out) the `out` parameter is not in the arg_map — it is
        # reached via maybe_get_output in the meta — so it is never built and the
        # zip above skips it.  The `.out` overload requires it, so create an empty
        # out tensor (the op resizes it; dtype defaults to the first input's).
        n = len(args)
        for i, sa in enumerate(schema.arguments):
            if i < n:
                continue
            if sa.kwarg_only and "Tensor" in str(sa.type) and sa.name not in kw:
                kw[sa.name] = torch.empty(0, dtype=first_tensor_dtype or torch.float32)
        return pos, kw
    except Exception:
        return list(args), {}


def _add_size_axioms(solver, named_vars: dict[str, Any],
                     max_size: int = _MAX_SIZE) -> None:
    """Bound tensor ndim and per-dim sizes to keep concretization tractable.

    Non-emptiness is NOT enforced here — it is decided per-sample by the diversity
    pins (`_diverse_pin_tiers`), which keep most samples non-empty while letting a
    fraction be deliberate empty / 0-dim edge cases.

    ``max_size`` is the per-dimension upper bound; the caller raises it for ops
    (e.g. packed-int4 matmul) whose validity needs dimensions larger than the
    default tractability cap.
    """
    def _bound_tv(tv: TensorVar) -> None:
        solver.add(tv.ndim <= _MAX_NDIM)
        for i in range(MAX_DIM):
            cell = Select(tv.sizes, IntVal(i))
            solver.add(cell >= 0, cell <= max_size)

    for var in named_vars.values():
        if isinstance(var, TensorVar):
            _bound_tv(var)
        elif isinstance(var, TensorListVar):
            for tv in var.tensors:
                _bound_tv(tv)
        elif isinstance(var, OptVar) and isinstance(var.value, TensorVar):
            _bound_tv(var.value)
        elif isinstance(var, IntArrayVar):
            # int[] elements range over [-MAX_NDIM, max_size]; bound up to
            # MAX_INT_ARRAY_LEN so length-6 int[] args (3-D padding) are fully
            # constrained, not just the first MAX_DIM elements.
            for i in range(MAX_INT_ARRAY_LEN):
                cell = Select(var.data, IntVal(i))
                solver.add(cell >= -_MAX_NDIM, cell <= max_size)


def _blocking_clause(named_vars: dict[str, Any], model) -> Any | None:
    """Return a clause that forces the next model to differ in at least one discrete dimension."""
    terms = []
    for var in named_vars.values():
        if isinstance(var, TensorVar):
            terms += [
                var.ndim  != model.eval(var.ndim,  model_completion=True),
                var.dtype != model.eval(var.dtype, model_completion=True),
            ]
        elif isinstance(var, (TensorListVar, ScalarListVar)):
            terms.append(var.length != model.eval(var.length, model_completion=True))
        elif isinstance(var, IntArrayVar):
            terms.append(var.length != model.eval(var.length, model_completion=True))
        elif isinstance(var, OptVar):
            terms.append(var.present != model.eval(var.present, model_completion=True))
        elif isinstance(var, (BoolRef,)):
            terms.append(var != model.eval(var, model_completion=True))
        elif isinstance(var, ArithRef) and var.is_int():
            terms.append(var != model.eval(var, model_completion=True))
    return Or(*terms) if terms else None


# Broad dtype pool for diversity pins.  Pinning a dtype an op doesn't support just
# makes that tier UNSAT, so the tiered fallback drops it — safe to include complex.
_DIVERSE_DTYPES = [INT32, INT64, FLOAT16, FLOAT32, FLOAT64, BFLOAT16, BOOL,
                   COMPLEX64, COMPLEX128]
_FLOAT_CODES = (FLOAT16, FLOAT32, FLOAT64, BFLOAT16)

# Generation is always diverse.  Inputs are mostly non-degenerate (so ops execute
# their real kernel), but a fraction are deliberately degenerate "edge" samples so
# the empty / 0-dim / inf-nan code paths still get covered.
_EDGE_PROB = 0.10
_EDGE_FLAVORS = ("empty", "scalar0", "infnan")


def _tensor_vars(named_vars: dict[str, Any]) -> list:
    out = []
    for v in named_vars.values():
        if isinstance(v, TensorVar):
            out.append(v)
        elif isinstance(v, OptVar) and isinstance(v.value, TensorVar):
            out.append(v.value)
    return out


def _scalar_vars(named_vars: dict[str, Any]) -> list:
    out = []
    for v in named_vars.values():
        if isinstance(v, ScalarVar):
            out.append(v)
        elif isinstance(v, OptVar) and isinstance(v.value, ScalarVar):
            out.append(v.value)
    return out


def _diverse_pin_tiers(named_vars, rng, dt, flavor):
    """Tiered pin lists (most → least specific) pushing the next model toward a fresh
    configuration.  `flavor` (None or an _EDGE_FLAVORS member) selects an edge shape.
    Tensor-shape diversity survives scalar conflicts (scalars dropped first)."""
    tvs = _tensor_vars(named_vars)

    def _nonempty(tv):
        ps = [tv.ndim >= 1]
        for i in range(MAX_DIM):
            ps.append(Implies(IntVal(i) < tv.ndim, Select(tv.sizes, IntVal(i)) >= 1))
        return ps

    def _exact(tv):
        nd = rng.randint(1, MAX_DIM)
        ps = [tv.ndim == nd]
        for i in range(nd):
            ps.append(Select(tv.sizes, IntVal(i)) == rng.randint(1, 4))
        return ps

    # ── shape pins (full = exact sizes; soft = just non-empty / edge) ──────────
    if flavor == "empty" and tvs:
        victim = rng.choice(tvs)
        nd = rng.randint(1, MAX_DIM)
        k = rng.randint(0, nd - 1)
        edge = [victim.ndim == nd, Select(victim.sizes, IntVal(k)) == 0]
        for tv in tvs:
            if tv is not victim:
                edge += _nonempty(tv)
        shape_full = shape_soft = edge
    elif flavor == "scalar0" and tvs:
        victim = rng.choice(tvs)
        edge = [victim.ndim == 0]
        for tv in tvs:
            if tv is not victim:
                edge += _nonempty(tv)
        shape_full = shape_soft = edge
    else:  # None or infnan → non-degenerate shapes
        shape_full = [p for tv in tvs for p in _exact(tv)]
        shape_soft = [p for tv in tvs for p in _nonempty(tv)]

    dtype_pins = [tv.dtype == dt for tv in tvs]

    # ── scalar / int / bool value pins ─────────────────────────────────────────
    scalar_pins = []
    for sv in _scalar_vars(named_vars):
        scalar_pins.append(sv.int_val == rng.randint(-8, 8))
        scalar_pins.append(sv.real_val == RealVal(rng.randint(-8, 8)))
        if flavor == "infnan":  # best-effort: only binds when the scalar is floating
            scalar_pins.append(sv.is_nan if rng.random() < 0.5 else sv.is_inf)
    for v in named_vars.values():
        if isinstance(v, ArithRef) and v.is_int():
            scalar_pins.append(v == rng.randint(0, 8))
        elif isinstance(v, BoolRef):
            scalar_pins.append(v == bool(rng.randint(0, 1)))

    return [
        shape_full + dtype_pins + scalar_pins,   # everything
        shape_soft + dtype_pins + scalar_pins,   # drop exact sizes (keep non-empty/edge)
        # Scalar diversity must survive a conflict on the TENSOR pins: in graph
        # generation a consumer's tensor port is externally pinned to a producer's
        # spec, so the diversity dtype/shape pins on that tensor go UNSAT. Keep the
        # scalar pins after peeling off dtype, then after peeling off shape too —
        # otherwise every consumer node falls through to the no-pin tier and its
        # scalar args collapse to Z3's default 0 (add.Scalar(x, 0, 0), clamp 0..0, …).
        shape_soft + scalar_pins,                # drop dtype (e.g. a pinned consumer dtype)
        scalar_pins,                             # drop all tensor pins — scalars still diversified
        shape_soft + dtype_pins,                 # drop scalars (scalar pins may be the conflict)
        dtype_pins,                              # drop shape
        [],                                      # last resort: no pins
    ]


def _diverse_model(solver, named_vars, rng, flavor):
    """Return a satisfying model biased toward a fresh (or edge) configuration, or
    None if the accumulated constraints are exhausted."""
    dt = rng.choice(_DIVERSE_DTYPES)
    for pins in _diverse_pin_tiers(named_vars, rng, dt, flavor):
        solver.push()
        for p in pins:
            solver.add(p)
        ok = solver.check() == sat
        if ok:
            m = solver.model()
            solver.pop()
            return m
        solver.pop()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test ATen op pass rate on Z3-generated valid inputs"
    )
    parser.add_argument("constraints_subfolder",
                        help="Path to constraints/<entry_sym>/ directory")
    parser.add_argument("--slices-dir", default=None,
                        help="(deprecated, ignored) slices are now self-contained "
                             "via get_params(); kept for CLI compatibility")
    parser.add_argument("--samples", "-n", type=int, default=1000)
    parser.add_argument("--pass-rate", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=50.0)
    parser.add_argument("--verbose", "-v", action="store_true")
    opts = parser.parse_args()

    subfolder = Path(opts.constraints_subfolder).resolve()
    if not subfolder.is_dir():
        print(f"ERROR: not a directory: {subfolder}", file=sys.stderr)
        return 2

    entry_sym = subfolder.name

    op, op_name, dispatch_key = find_op_for_symbol(entry_sym)
    if op is None:
        print(f"ERROR: no ATen op found for symbol: {entry_sym[:80]}", file=sys.stderr)
        return 2

    print(f"Operator  : {op_name}  (dispatch_key={dispatch_key})", flush=True)

    # ── load constraint files ──────────────────────────────────────────────────
    named_params: list[tuple[str, str]] | None = None
    named_vars: dict[str, Any] = {}
    bad_constraints: list[list] = []
    slice_axioms: list[list] = []  # unconditional linking axioms (not negated)
    n_stub = n_ok = n_err = 0

    for py_path in sorted(subfolder.glob("slice_*.py")):
        try:
            spec = importlib.util.spec_from_file_location("_slice", py_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as exc:
            n_err += 1
            if opts.verbose:
                print(f"  SKIP {py_path.name}: load error: {exc}", flush=True)
            continue

        # Build named_params from the slice's own get_params() on the first slice.
        # The emitter bakes the full ordered (name, model_type) arg list into every
        # slice, so the harness is self-contained — no slices_ir JSON needed.
        if named_params is None:
            if not hasattr(mod, "get_params"):
                n_err += 1
                if opts.verbose:
                    print(f"  SKIP {py_path.name}: no get_params() — regenerate slice",
                          flush=True)
                continue
            named_params = mod.get_params()

        # Build named_vars from named_params the first time.
        # Z3 variables are interned by name, so Int('dim') here is the same
        # object as Int('dim') inside get_constraints() — no identity mismatch.
        if not named_vars:
            named_vars = {
                name: _ARG_BUILDERS[mt](name)
                for name, mt in named_params
                if mt in _ARG_BUILDERS
            }

        try:
            clist = _flatten(mod.get_constraints())
        except Exception as exc:
            n_err += 1
            if opts.verbose:
                print(f"  SKIP {py_path.name}: constraints error: {exc}", flush=True)
            continue

        # Axioms must be loaded regardless of whether constraints are empty.
        # iter_config from_meta slices have 0 constraints but carry all the
        # broadcastability / same-dtype / canCast axioms.
        if hasattr(mod, "get_axioms"):
            try:
                axioms = _flatten(mod.get_axioms())
                if axioms:
                    slice_axioms.append(axioms)
            except Exception:
                pass

        if not clist:
            n_stub += 1
            continue

        bad_constraints.append(clist)
        n_ok += 1

    print(f"Slices    : {n_ok} usable  ({n_stub} stub, {n_err} error)", flush=True)

    if not named_params or not bad_constraints:
        print("ERROR: no usable constraints loaded", file=sys.stderr)
        return 2

    # ── Phase 1: enumerate valid inputs via Z3 ─────────────────────────────────
    print(f"\nPhase 1 — generating up to {opts.samples} valid inputs ...", flush=True)

    solver = _make_solver()
    _add_domain_axioms(solver, named_vars)
    # Packed-int4 matmul needs K>=32 / N>=16 — far above the default size cap.
    # Raise the per-dim bound only for that op family so other ops keep the small
    # tractable bound.
    _sz = 256 if (op_name or "").startswith("_weight_int4pack_mm") else _MAX_SIZE
    _add_size_axioms(solver, named_vars, max_size=_sz)
    # Add unconditional linking axioms directly (not negated).
    for axioms in slice_axioms:
        flat = _flatten(axioms)
        if flat:
            solver.add(*flat)
    for clist in bad_constraints:
        flat = _flatten(clist)
        if flat:
            solver.add(Not(And(*flat)))

    # Diverse generation: random shape/dtype/scalar variation via solver pins +
    # random tensor data.  ~10% of samples are deliberate edge cases.  Seeded for
    # reproducibility.
    rng = random.Random(int(os.environ.get("DIVERSE_SEED", "12648430")))  # 0xC0FFEE; override to stress-test

    concrete_list: list[Any] = []
    for i in range(opts.samples):
        flavor = rng.choice(_EDGE_FLAVORS) if rng.random() < _EDGE_PROB else None
        m = _diverse_model(solver, named_vars, rng, flavor)
        if m is None:
            print(f"  Z3: search space exhausted after {i} samples", flush=True)
            break
        data_mode = "infnan" if flavor == "infnan" else "random"
        try:
            concrete = build_call_args(named_params, named_vars, m, rng=rng,
                                       data_mode=data_mode, op_name=op_name)
        except Exception as exc:
            if opts.verbose:
                print(f"  sample {i}: concretize error: {exc}", flush=True)
            concrete = None
        concrete_list.append(concrete)
        if (i + 1) % 200 == 0:
            print(f"  enumerated {i+1}/{opts.samples} ...", flush=True)
        bc = _blocking_clause(named_vars, m)
        if bc is None:
            break
        solver.add(bc)

    print(f"  Generated {len(concrete_list)} inputs", flush=True)
    if not concrete_list:
        print("ERROR: Z3 produced no inputs", file=sys.stderr)
        return 2

    # ── Phase 2: run operator on each input ────────────────────────────────────
    print(f"\nPhase 2 — running operator on {len(concrete_list)} inputs ...", flush=True)

    n_pass = n_fail = n_skip = 0
    for i, concrete in enumerate(concrete_list):
        if concrete is None:
            n_skip += 1
            continue
        pos_args, kw_args = _split_by_schema(op, concrete.args)
        from functools import partial
        call_op = partial(op, **kw_args) if kw_args else op
        triggered, error = _call_op_forked(call_op, pos_args, timeout=opts.timeout)
        if triggered is False:
            n_pass += 1
            if opts.verbose:
                print(f"  [{i}] PASS", flush=True)
        elif triggered is True:
            n_fail += 1
            if opts.verbose:
                shapes = {k: (list(v.shape) if hasattr(v, "shape") else repr(v))
                          for k, v in (concrete.witness or {}).items()}
                print(f"  [{i}] FAIL  shapes={shapes}  error={error!r}", flush=True)
        else:
            n_skip += 1
            if opts.verbose:
                print(f"  [{i}] SKIP  {error or ''}", flush=True)

    # ── Report ─────────────────────────────────────────────────────────────────
    n_total = len(concrete_list)
    n_tested = n_pass + n_fail
    pass_rate = n_pass / n_tested if n_tested > 0 else 0.0

    print(f"\n── Results {'─' * 45}")
    print(f"  Generated : {n_total}")
    print(f"  Pass      : {n_pass:6d}  ({100*n_pass/n_total:.1f}%)")
    print(f"  Fail      : {n_fail:6d}  ({100*n_fail/n_total:.1f}%)")
    print(f"  Skip      : {n_skip:6d}  ({100*n_skip/n_total:.1f}%)")
    print(f"  Tested    : {n_tested}")
    if n_tested > 0:
        print(f"  Pass rate : {100*pass_rate:.2f}%  (threshold {100*opts.pass_rate:.1f}%)")
    print()

    if n_tested == 0:
        print("RESULT: INCONCLUSIVE (all samples skipped)", flush=True)
        return 2
    if pass_rate >= opts.pass_rate:
        print(f"RESULT: PASS  {100*pass_rate:.2f}% >= {100*opts.pass_rate:.1f}%", flush=True)
        return 0
    print(f"RESULT: FAIL  {100*pass_rate:.2f}% < {100*opts.pass_rate:.1f}%", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
