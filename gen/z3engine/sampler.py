"""sampler.py — diverse valid-input sampler over the Z3 constraint model.

Library (no CLI). Given an op's Z3 solver (assembled in gen/ops/opnode.py),
`_diverse_model` returns ONE satisfying model biased toward a fresh, diverse
configuration; `_diverse_pin_tiers` builds the tiered diversity pins (shape / dtype /
scalar) that produce that bias. The corpus generator calls this via
`gen/ops/opnode.py::OpNode.generate` to materialize each node's inputs.

Sampling is always diverse: random shape/dtype/scalar variation driven through the
solver, with random tensor data. Most samples are non-degenerate so ops run their real
kernel; a small fraction stay degenerate (empty / 0-dim) when an op requires it or via
`_DEGENERATE_SAMPLE_PROB`.
"""

from __future__ import annotations

from typing import Any

from z3 import Implies, IntVal, RealVal, Select, BoolRef, ArithRef, sat

from mobile.gen.z3engine.model import (
    TensorVar, IntArrayVar, OptVar, ScalarVar,
    MAX_DIM,
    MAX_INT_ARRAY_LEN,
    INT32, INT64, FLOAT16, FLOAT32, FLOAT64, BFLOAT16, BOOL, COMPLEX64, COMPLEX128,
)


# Broad dtype pool for diversity pins.  Pinning a dtype an op doesn't support just
# makes that tier UNSAT, so the tiered fallback drops it — safe to include complex.
_DIVERSE_DTYPES = [INT32, INT64, FLOAT16, FLOAT32, FLOAT64, BFLOAT16, BOOL,
                   COMPLEX64, COMPLEX128]

# Output-shape int[] args (broadcast/expand/reshape/factory `size`, etc.) whose elements ARE
# an output tensor's dims, so a 0 element → an empty/degenerate output tensor. We bias these
# >= 1 (see _shape_array_pins) so an unconstrained shape can't collapse to [0, 0, ...]. Every
# OTHER int[] arg (dim/pad/stride/dilation/kernel_size/output_padding) legitimately holds 0 or
# negatives and is left untouched. Matched by arg NAME because our model collapses all int[]
# to one type, and a blanket >= 1 would wrongly forbid negative dims / zero padding.
_SHAPE_ARRAY_ARGS = frozenset({
    "size", "sizes", "shape", "output_size", "input_size", "input_sizes", "normalized_shape",
})
_DEGENERATE_SAMPLE_PROB = 0.01   # fraction of samples left fully shape-unpinned → may be
                                 # degenerate (empty / 0-dim tensor, or zero-shape int[] arg)


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


def _shape_array_pins(named_vars: dict[str, Any]) -> list:
    """Pins forcing every SHAPE-typed int[] arg's elements to be >= 1, so an
    unconstrained shape can't collapse to [0, 0, ...] (an empty output tensor).
    Non-shape int[] args (dim / pad / stride / ...) are left alone — 0 and negative
    values are legitimate there."""
    pins = []
    for name, var in named_vars.items():
        if name not in _SHAPE_ARRAY_ARGS:
            continue
        # `int[]?` args are wrapped in an OptVar; unwrap to the inner IntArrayVar.
        iav = var.value if isinstance(var, OptVar) else var
        if not isinstance(iav, IntArrayVar):
            continue
        # Constrain each element slot that's actually within the array's length.
        for i in range(MAX_INT_ARRAY_LEN):
            pins.append(Implies(IntVal(i) < iav.length, iav[i] >= 1))
    return pins


def _diverse_pin_tiers(named_vars, rng, dt):
    """Build a priority list of pin-sets (most → least specific) biasing the next model
    toward a fresh, diverse configuration. `_diverse_model` tries them in order and keeps
    the first SAT one, so an over-constrained tier (a dtype the op rejects, or a tensor
    port already pinned to a producer) is simply skipped for a looser one."""
    tvs = _tensor_vars(named_vars)

    # Per-tensor shape pins, two strengths:
    def _nonempty(tv):   # ndim >= 1 and every used dim >= 1
        ps = [tv.ndim >= 1]
        for i in range(MAX_DIM):
            ps.append(Implies(IntVal(i) < tv.ndim, Select(tv.sizes, IntVal(i)) >= 1))
        return ps

    def _exact(tv):      # a specific random ndim, each dim a specific size in 1..4
        nd = rng.randint(1, MAX_DIM)
        ps = [tv.ndim == nd]
        for i in range(nd):
            ps.append(Select(tv.sizes, IntVal(i)) == rng.randint(1, 4))
        return ps

    # ~99% of the time, bias toward a NON-degenerate sample: pin every tensor to a real
    # shape (exact / non-empty) AND keep output-shape int[] args (size/shape/...) >= 1 so
    # they can't collapse to [0, 0, ...]. The other ~_DEGENERATE_SAMPLE_PROB of the time we
    # add NO shape pins, leaving the solver free to pick a degenerate sample (empty / 0-dim
    # tensor, zero-shape) — deliberate edge-case coverage. (When an op *requires* a 0-size
    # shape, the >= 1 pins go UNSAT and the tier ladder falls through to a pin-free tier
    # anyway, so required-degenerate ops still work even in the 99% branch.)
    if rng.random() >= _DEGENERATE_SAMPLE_PROB:
        shape_full = [p for tv in tvs for p in _exact(tv)]
        shape_soft = [p for tv in tvs for p in _nonempty(tv)]
        shape_arr = _shape_array_pins(named_vars)
        shape_full += shape_arr
        shape_soft += shape_arr
    else:
        shape_full = shape_soft = []

    dtype_pins = [tv.dtype == dt for tv in tvs]

    scalar_pins = []
    for sv in _scalar_vars(named_vars):
        scalar_pins.append(sv.int_val == rng.randint(-8, 8))
        scalar_pins.append(sv.real_val == RealVal(rng.randint(-8, 8)))
    for v in named_vars.values():
        if isinstance(v, ArithRef) and v.is_int():
            scalar_pins.append(v == rng.randint(0, 8))
        elif isinstance(v, BoolRef):
            scalar_pins.append(v == bool(rng.randint(0, 1)))

    # The ladder peels off shape (exact→soft→none), then dtype, keeping scalar pins
    # longest: a consumer's tensor port is often externally pinned to a producer, so its
    # shape/dtype pins go UNSAT — but its scalar args must still be diversified (else they
    # collapse to Z3's default 0, e.g. add.Scalar(x, 0, 0), clamp 0..0).
    return [
        shape_full + dtype_pins + scalar_pins,
        shape_soft + dtype_pins + scalar_pins,
        shape_soft + scalar_pins,
        scalar_pins,
        shape_soft + dtype_pins,
        dtype_pins,
        [],
    ]


def _diverse_model(solver, named_vars, rng):
    """Return a satisfying model biased toward a fresh, diverse configuration, or None
    if every pin tier (down to no pins) is UNSAT."""
    dt = rng.choice(_DIVERSE_DTYPES)
    for pins in _diverse_pin_tiers(named_vars, rng, dt):
        solver.push()
        for p in pins:
            solver.add(p)
        if solver.check() == sat:
            m = solver.model()
            solver.pop()
            return m
        solver.pop()
    return None
