"""Z3 model -> Python scalar evaluators.

Thin wrappers around model.eval that coerce a Z3 result into a concrete int /
bool / float, with safe defaults when the expression is free or non-numeric.
"""

from __future__ import annotations

from fractions import Fraction

from z3 import BoolRef, IntNumRef, RatNumRef, is_true


def _eval(model, expr):
    return model.eval(expr, model_completion=True)


def _eval_int(model, expr, default: int = 0) -> int:
    try:
        val = _eval(model, expr)
        if isinstance(val, IntNumRef):
            return int(val.as_long())
        return int(str(val))
    except Exception:
        return default


def _eval_bool(model, expr, default: bool = False) -> bool:
    try:
        val = _eval(model, expr)
        if isinstance(val, BoolRef):
            return is_true(val)
        return bool(val)
    except Exception:
        return default


def _eval_bound(model, expr):
    """Return a float bound only if `expr` is actually constrained, else None.

    Uses model_completion=False so a FREE variable (one no slice constrained) comes
    back non-numeric → None → the caller applies no clamp and the tensor keeps its
    full random range.  A slice-declared bound (e.g. std.data_lo >= 0) is assigned a
    value in the model → returned as a float → the caller clamps the data to it.
    Generic: the concretizer never needs to know which op/arg the bound came from."""
    try:
        val = model.eval(expr, model_completion=False)
        if isinstance(val, RatNumRef):
            return float(Fraction(val.numerator_as_long(), val.denominator_as_long()))
        if isinstance(val, IntNumRef):
            return float(val.as_long())
        return None  # free variable — non-numeric
    except Exception:
        return None


def _eval_float(model, expr, default: float = 0.0) -> float:
    try:
        val = _eval(model, expr)
        if isinstance(val, RatNumRef):
            return float(Fraction(val.numerator_as_long(), val.denominator_as_long()))
        return float(str(val).replace("?", ""))
    except Exception:
        return default
