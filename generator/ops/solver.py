"""solver.py — assemble an op's Z3 precondition, and draw diverse samples from it.

Assembly turns an OpConstraints into a solver that describes exactly the valid calls:
per-variable domain axioms, the generation bounds (tensors stay small), the op's own
axioms, and the negation of every error path it can reach.

Sampling then pulls a *varied* witness out of that solver. Left alone, Z3 returns the same
minimal model every time — all-zero shapes, the lowest dtype code — so the corpus would be
thousands of copies of one degenerate call. `diverse_model` biases it with random pins, and
drops them a tier at a time when they conflict with the op's real constraints.

Everything here is per-OP: `build_solver` takes one OpConstraints and `diverse_model` one
op's variables. There is no graph-level solve — the builder grows a DAG by solving each
node alone, pinned against the CONCRETE tensor a producer already returned, so no two
nodes are ever in the same Z3 problem.
"""

from __future__ import annotations

import random
from typing import Any

from z3 import (
    And, ArithRef, BoolRef, Implies, IntVal, Not, RealVal, Select, Solver, sat,
)

from mobile.generator.constraints.model import (
    MAX_DIM, MAX_INT_ARRAY_LEN, VALID_DTYPE_CODES,
    IntArrayVar, OptVar, ScalarVar, TensorListVar, TensorVar,
    compatible_dtypes,
)

# Z3 wall-clock budget per check(). A precondition that needs longer is not worth the
# worker time — the op simply fails to generate this round.
TIMEOUT_MS = 10_000

# Generated tensors stay small on purpose: the point is operator coverage, not size
# coverage, and every element costs eager time, .pte bytes, and device transfer.
MAX_SIZE = 4

# Dtypes to bias a sample toward: everything the model can represent. Pinning one an op
# rejects just makes that tier UNSAT and the ladder drops it, so listing a dtype here is
# safe even where it is rarely valid. Deliberately NOT narrowed to what the reference
# runtime handles well — an op is only ever tested on the dtypes we generate for it, so
# narrowing here would quietly decide that whole dtype families are correct by never
# asking. Kernels that have no implementation for a dtype report it cleanly at run time.
_DIVERSE_DTYPES = sorted(VALID_DTYPE_CODES)

# int[] args whose elements ARE an output tensor's dims, so a 0 element yields an empty
# result. Biased >= 1 below. Every OTHER int[] (dim/pad/stride/dilation/kernel_size)
# legitimately holds 0 or negatives, and the model collapses all int[] to one type, so
# this has to be matched by argument NAME.
_SHAPE_ARRAY_ARGS = frozenset({
    "size", "sizes", "shape", "output_size", "input_size", "input_sizes",
    "normalized_shape",
})

# Fraction of samples left shape-unpinned, free to come back degenerate (empty or 0-dim).
# Deliberate edge coverage; kept small so it barely dents yield.
DEGENERATE_SAMPLE_PROB = 0.01


def _flatten(items: list) -> list:
    """Flatten nested lists — Z3 never accepts a list where it wants an expression."""
    out = []
    for item in items:
        out.extend(_flatten(item)) if isinstance(item, list) else out.append(item)
    return out


def _tensors_of(var) -> list[TensorVar]:
    """Every TensorVar reachable from one argument variable (unwrapping optionals and
    tensor lists), so bounds can be applied uniformly."""
    if isinstance(var, TensorVar):
        return [var]
    if isinstance(var, TensorListVar):
        return list(var.tensors)
    if isinstance(var, OptVar) and isinstance(var.value, TensorVar):
        return [var.value]
    return []


def _scalars_of(var) -> list[ScalarVar]:
    if isinstance(var, ScalarVar):
        return [var]
    if isinstance(var, OptVar) and isinstance(var.value, ScalarVar):
        return [var.value]
    return []


def _domain_axioms(variables: dict[str, Any]) -> list:
    """Each variable's own well-formedness axioms, plus pairwise dtype compatibility.

    The pairwise part keeps Z3 from proposing a dtype combination that
    c10::promoteTypes would throw on (Float8 with anything else, wide-unsigned with a
    non-float) — those inputs would fail at the call for a reason that has nothing to do
    with the op under test.
    """
    out = []
    for var in variables.values():
        if hasattr(var, "axioms"):
            out.extend(var.axioms())
    dtypes = [v.dtype for v in variables.values() if isinstance(v, TensorVar)]
    for i in range(len(dtypes)):
        for j in range(i + 1, len(dtypes)):
            out.append(compatible_dtypes(dtypes[i], dtypes[j]))
    return out


def _generation_bounds(variables: dict[str, Any], fixed_array_len: dict[str, int]) -> list:
    """Bounds that keep a sample generatable: small tensors, sane int[] lengths and
    elements.

    Note what is NOT here: any restriction on dtype. The model already limits each tensor
    to VALID_DTYPE_CODES, and narrowing further to what the reference runtime handles well
    would mean never testing the dtypes it handles badly — which is the opposite of the
    job. An unsupported dtype surfaces as a clean unhandled-dtype report at run time.
    """
    out = []
    for var in variables.values():
        for tv in _tensors_of(var):
            out.append(tv.ndim <= MAX_DIM)
            for i in range(MAX_DIM):
                cell = Select(tv.sizes, IntVal(i))
                out.extend([cell >= 0, cell <= MAX_SIZE])

        if isinstance(var, IntArrayVar):
            # A fixed-arity `int[N]` (e.g. reflection_pad3d's `SymInt[6] padding`) loses
            # its N in the model, which collapses every int[] to one type. Left free, the
            # length roams [0, MAX_INT_ARRAY_LEN] and almost never lands on N, starving
            # the op. Pin it when N fits the element budget.
            n_fixed = fixed_array_len.get(var.name)
            pinned = n_fixed is not None and n_fixed <= MAX_INT_ARRAY_LEN
            if pinned:
                out.append(var.length == n_fixed)
            # Bounding the ELEMENTS is not optional: an unbounded `size`/`shape` arg
            # (factory ops, reshape, empty_permuted) lets the concretizer try to allocate
            # a giant tensor and take the worker out with it.
            for i in range(n_fixed if pinned else MAX_INT_ARRAY_LEN):
                cell = Select(var.data, IntVal(i))
                out.extend([cell >= -MAX_DIM, cell <= MAX_SIZE])
    return out


def build_solver(constraints, fixed_array_len: dict[str, int] | None = None,
                 pins: list | None = None) -> Solver:
    """An OpConstraints → a solver whose models are valid calls to that op.

    `constraints.bad` holds, per slice, the conditions under which the op RAISES; a valid
    call is the negation of each. `pins` are extra constraints from the caller — a port
    tied to a producer's shape/dtype, or the sampler's diversity bias.

    A target runtime's extra requirements arrive already merged into `constraints.axioms`
    (see targets.load_for), so they are asserted like any other axiom rather
    than offered as a relaxable pin — they are not a diversity preference the ladder may
    trade away, but the difference between a .pte that runs and one the runtime refuses.
    """
    solver = Solver()
    solver.set("timeout", TIMEOUT_MS)
    solver.add(*_domain_axioms(constraints.vars))
    solver.add(*_generation_bounds(constraints.vars, fixed_array_len or {}))
    for axioms in constraints.axioms:
        solver.add(*_flatten(axioms))
    for error_path in constraints.bad:
        solver.add(Not(And(*_flatten(error_path))))
    if pins:
        solver.add(*pins)
    return solver


def _shape_array_pins(variables: dict[str, Any]) -> list:
    """Force every SHAPE-typed int[] element >= 1, so an unconstrained `size`/`shape`
    cannot collapse to [0, 0, ...] and produce an empty output."""
    pins = []
    for name, var in variables.items():
        if name not in _SHAPE_ARRAY_ARGS:
            continue
        array = var.value if isinstance(var, OptVar) else var
        if not isinstance(array, IntArrayVar):
            continue
        for i in range(MAX_INT_ARRAY_LEN):
            pins.append(Implies(IntVal(i) < array.length, array[i] >= 1))
    return pins


def _pin_tiers(variables: dict[str, Any], rng: random.Random, dtype: int) -> list[list]:
    """Diversity pins, most specific first. `diverse_model` keeps the first tier that is
    still SAT, so an over-constrained tier — a dtype the op rejects, or a tensor port
    already pinned to a producer — is simply skipped for a looser one."""
    tensors = [tv for var in variables.values() for tv in _tensors_of(var)]

    def exact(tv):          # one specific ndim, each dim a specific small size
        ndim = rng.randint(1, MAX_DIM)
        return [tv.ndim == ndim] + [Select(tv.sizes, IntVal(i)) == rng.randint(1, MAX_SIZE)
                                    for i in range(ndim)]

    def nonempty(tv):       # at least 1-D, and no used dim is 0
        return [tv.ndim >= 1] + [Implies(IntVal(i) < tv.ndim, Select(tv.sizes, IntVal(i)) >= 1)
                                 for i in range(MAX_DIM)]

    if rng.random() >= DEGENERATE_SAMPLE_PROB:
        array_pins = _shape_array_pins(variables)
        shape_exact = [p for tv in tensors for p in exact(tv)] + array_pins
        shape_soft = [p for tv in tensors for p in nonempty(tv)] + array_pins
    else:
        shape_exact = shape_soft = []       # let this one come back degenerate

    dtype_pins = [tv.dtype == dtype for tv in tensors]

    scalar_pins = []
    for var in variables.values():
        for sv in _scalars_of(var):
            scalar_pins.append(sv.int_val == rng.randint(-8, 8))
            scalar_pins.append(sv.real_val == RealVal(rng.randint(-8, 8)))
    for var in variables.values():
        if isinstance(var, ArithRef) and var.is_int():
            scalar_pins.append(var == rng.randint(0, 8))
        elif isinstance(var, BoolRef):
            scalar_pins.append(var == bool(rng.randint(0, 1)))

    # Shape peels off first, then dtype; scalar pins are held longest. A consumer's tensor
    # port is often already pinned to a producer, so its shape/dtype pins go UNSAT — but
    # its scalar arguments must still vary, or they collapse to Z3's default 0 and the
    # corpus fills up with add(x, 0, 0) and clamp(0, 0).
    return [
        shape_exact + dtype_pins + scalar_pins,
        shape_soft + dtype_pins + scalar_pins,
        shape_soft + scalar_pins,
        scalar_pins,
        shape_soft + dtype_pins,
        dtype_pins,
        [],
    ]


def diverse_model(solver: Solver, variables: dict[str, Any], rng: random.Random):
    """A satisfying model biased toward a fresh configuration, or None if even the
    unpinned solver is UNSAT."""
    dtype = rng.choice(_DIVERSE_DTYPES)
    for pins in _pin_tiers(variables, rng, dtype):
        solver.push()
        solver.add(*pins)
        if solver.check() == sat:
            model = solver.model()
            solver.pop()
            return model
        solver.pop()
    return None
