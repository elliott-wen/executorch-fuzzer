"""op.py — one generatable operator: its callable, its precondition, and valid inputs.

An Op ties together the three lower layers — the constraints loaded for its symbol, the
solver built from them, and the concretizer that turns a solved model into real tensors —
and exposes the two things the graph builder needs: generate a valid call, and pin a port
to a producer's output so the next node can be wired onto it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from z3 import IntVal, Select, is_const

from mobile.generator.concretize import build_call_args, dtype_code_for
from mobile.generator.constraints import OpConstraints
from mobile.generator.constraints.model import TensorVar
from mobile.generator.ops import solver as _solver


def _referenced_names(expressions) -> set[str]:
    """Every constant/variable name appearing anywhere in a set of Z3 expressions
    (one walk over the DAG, memoized on node id)."""
    names: set[str] = set()
    seen: set[int] = set()

    def walk(expr):
        if expr.get_id() in seen:
            return
        seen.add(expr.get_id())
        if is_const(expr):
            names.add(expr.decl().name())
        for child in expr.children():
            walk(child)

    for expr in expressions:
        if hasattr(expr, "children"):
            walk(expr)
    return names


def _as_float(model, expr) -> float | None:
    """Evaluate a Z3 numeral to a float, or None if it isn't numeric."""
    try:
        value = model.eval(expr, model_completion=True)
        if hasattr(value, "as_long"):
            return float(value.as_long())
        if hasattr(value, "numerator_as_long"):
            return value.numerator_as_long() / value.denominator_as_long()
        return float(str(value).replace("?", ""))
    except Exception:
        return None


@dataclass
class Op:
    """One operator the generator can emit calls to."""

    symbol: str
    op_name: str
    fn: Any                                   # the resolved torch.ops.aten callable
    constraints: OpConstraints
    fixed_array_len: dict[str, int] = field(default_factory=dict)

    # Derived once at construction — value_ports walks the whole constraint DAG, which is
    # far too expensive to repeat on every generate().
    tensor_ports: list[str] = field(init=False)
    value_ports: set[str] = field(init=False)

    def __post_init__(self) -> None:
        # Positional Tensor parameters a producer can be wired into. `out` is excluded:
        # it is a destination, not an input.
        self.tensor_ports = [n for n, t in self.params if t == "Tensor" and n != "out"]
        # Ports whose DATA is constrained, not just shape and dtype (index / target /
        # probability arguments, bounded via `<port>.data_lo`/`.data_hi`). A fresh leaf
        # for such a port is clamped into range by the concretizer, but a producer wired
        # into it is not — so the graph builder must insert a clamp, and this set is the
        # only way it can know to.
        names = _referenced_names(e for path in self.constraints.bad for e in path)
        self.value_ports = {p for p in self.tensor_ports
                            if f"{p}.data_lo" in names or f"{p}.data_hi" in names}

    @property
    def params(self) -> list[tuple[str, str]]:
        return self.constraints.params

    @property
    def vars(self) -> dict[str, Any]:
        return self.constraints.vars

    @property
    def label(self) -> str:
        return self.op_name or self.symbol[:24]

    def solver(self, pins: list | None = None):
        return _solver.build_solver(self.constraints, self.fixed_array_len, pins)

    def pin_for(self, port: str, tensor) -> list:
        """Constraints forcing `port` to match a concrete tensor's shape and dtype — how a
        consumer is tied to the producer it will read from."""
        var = self.vars[port]
        pins = [var.ndim == tensor.dim()]
        pins += [Select(var.sizes, IntVal(i)) == tensor.size(i) for i in range(tensor.dim())]
        code = dtype_code_for(tensor.dtype)
        if code is not None:
            pins.append(var.dtype == code)
        return pins

    def generate(self, rng, pins: list | None = None):
        """Concrete arguments for one valid call, or None if the precondition (with these
        pins) has no model.

        The result carries `.ranges`: {port: (lo, hi)} for each value-constrained port,
        matching the clamp the concretizer applied to that leaf, so the graph builder can
        clamp a wired producer the same way.
        """
        model = _solver.diverse_model(self.solver(pins), self.vars, rng)
        if model is None:
            return None
        args = build_call_args(self.params, self.vars, model, rng=rng,
                               data_mode="random", op_name=self.op_name)
        ranges = {}
        for port in self.value_ports:
            var = self.vars.get(port)
            if not isinstance(var, TensorVar):
                continue
            lo, hi = _as_float(model, var.data_lo), _as_float(model, var.data_hi)
            if lo is None or hi is None:
                continue
            lo, hi = math.ceil(lo), math.floor(hi)
            if hi >= lo:                    # an empty range means "leave it a leaf"
                ranges[port] = (lo, hi)
        args.ranges = ranges
        return args
