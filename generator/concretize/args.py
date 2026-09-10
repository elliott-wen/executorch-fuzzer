"""Top-level entry point: build concrete call args in signature order.

build_call_args walks a PyTorch op's named params, materialises each via the builders,
and applies special-arg coercion. Returns ConcreteArgs (args + witness + errors).

It reports what the solver decided and repairs nothing: an argument that violates the
op's precondition means the precondition was not constrained, and the fix belongs in the
constraint, not in a post-solve patch-up here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .builders import _build_value
from .coerce import _coerce_special_arg


@dataclass
class ConcreteArgs:
    args: list[Any] = field(default_factory=list)
    witness: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


# Matches the float[]/bool[] list model-types built as fixed defaults:
#   float[] / bool[] (dynamic), float[]? / bool[]? (optional), bool[3] (fixed size)
_LISTARG_RE = re.compile(r"^(float|bool)\[(\d*)\](\?)?$")


def build_call_args(
    named_params: list[tuple[str, str]],
    named_vars: dict[str, Any],
    model,
    rng=None,
    data_mode: str = "random",
    op_name: str | None = None,
) -> ConcreteArgs:
    """Build concrete args in PyTorch signature order from a Z3 model.

    When ``rng`` is provided, tensor element data is randomized (diversity mode);
    otherwise tensors are zero-filled.  ``data_mode`` ("random"/"infnan") selects
    the fill style.  Scalar/int values come from the model regardless (their
    diversity is driven through the solver, keeping constraints satisfied)."""
    result = ConcreteArgs()
    for name, typ in named_params:
        # float[] / bool[] list args (model has no var for these — they carry no
        # solver constraints; build a valid fixed default by construction):
        #   float[]? / bool[]? (e.g. histogram `range`, `addends`)  → None (absent)
        #   bool[N]  (std::array<bool,N>, e.g. `output_mask`)       → [True]*N
        #   bool[] / float[] (dynamic, required — rare)             → small default
        lm = _LISTARG_RE.match(typ)
        if lm is not None:
            elem, nstr, opt = lm.group(1), lm.group(2), lm.group(3)
            if opt == "?":
                value = None
            elif nstr:
                n = int(nstr)
                value = [True] * n if elem == "bool" else [0.0] * n
            else:
                value = [True] if elem == "bool" else []
            value = _coerce_special_arg(name, value, op_name)
            result.args.append(value)
            result.witness[name] = value
            continue
        var = named_vars.get(name)
        if var is None:
            result.errors.append(f"missing symbolic variable for {name}: {typ}")
            result.args.append(None)
            result.witness[name] = None
            continue
        try:
            value = _build_value(typ, var, model, rng, data_mode, arg_name=name,
                                 op_name=op_name)
            value = _coerce_special_arg(name, value, op_name)
        except Exception as exc:
            result.errors.append(f"{name}: {typ}: {exc}")
            value = None
        result.args.append(value)
        result.witness[name] = value

    return result
