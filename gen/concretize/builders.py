"""Per-type value builders: materialise concrete PyTorch values from Z3 witnesses.

One builder per model var type (tensor, scalar, int[], lists, optionals); _build_value
dispatches on the native type string. Tensor element data is NOT tracked by the model,
so randomizing it here is sound (it only affects value-dependent op behavior).
"""

from __future__ import annotations

from typing import Any

from z3 import Select, IntVal

from mobile.gen.z3engine.model import (
    TensorVar, ScalarVar, IntArrayVar, OptVar, TensorListVar, ScalarListVar,
    GeneratorStub,
    MAX_DIM, MAX_TENS, MAX_INT_ARRAY_LEN,
    INT64, FLOAT32,
    COMPLEX32, COMPLEX64, COMPLEX128,
    BOOL,
    FLOATING_DTYPES,
)

from .dtypes import _simplify_dtype, _torch_dtypes
from .evals import _eval_int, _eval_bool, _eval_bound, _eval_float
from .enums import _build_string_enum


def _clamp_int(value: int, lo: int, hi: int) -> int:
    if value < lo:
        return lo
    if value <= hi:
        return value
    # Use modular reduction to preserve remainder (e.g. oddness, divisibility)
    # while mapping into [lo, hi].  Plain truncation to hi loses these properties.
    return lo + (value - lo) % (hi - lo + 1)


_MAX_ELEMENTS = 1024  # guard against OOM from large Z3 witnesses (e.g. 32^8)


def _safe_sizes(sizes: list[int]) -> list[int]:
    """Clamp per-dim sizes and cap total elements to _MAX_ELEMENTS."""
    out = [_clamp_int(s, 0, 32) for s in sizes]
    total = 1
    for i, s in enumerate(out):
        budget = max(1, _MAX_ELEMENTS // total)
        out[i] = min(s, budget)
        total *= max(out[i], 1)
    return out


def _randomize_(t, rng, mode: str = "random") -> None:
    """Fill a freshly-allocated tensor in place with random values appropriate to
    its dtype.  Tensor element data is NOT tracked by the Z3 model, so randomizing
    it here is sound — it only affects value-dependent op behavior, never the
    shape/dtype constraints.  Best-effort: dtypes that don't support the fill
    (e.g. float8) are left zeroed.

    mode="infnan" sprinkles inf/-inf/nan into floating/complex tensors (edge
    coverage for NaN-propagation / guard paths); ignored for integer/bool dtypes."""
    import math
    import torch
    try:
        torch.manual_seed(rng.randint(0, 2**31 - 1))
        dt = t.dtype
        if dt == torch.bool:
            t.random_(0, 2)
        elif dt.is_complex or dt.is_floating_point:
            src = torch.randn(t.shape, dtype=dt)
            if mode == "infnan" and src.numel():
                flat = src.view(-1)
                for j in range(flat.numel()):
                    if rng.random() < 0.4:
                        flat[j] = rng.choice((math.inf, -math.inf, math.nan))
            t.copy_(src)
        else:  # integer dtypes — small non-negative range, valid for uint8 too
            t.random_(0, 16)
    except Exception:
        pass  # leave the zero-fill (e.g. float8_* has no random_/normal_)


def _build_tensor(tv: TensorVar, model, rng=None, data_mode: str = "random") -> Any:
    """Build a concrete dense (strided) tensor from a Z3 model witness.

    When ``rng`` is provided, the tensor is filled with random values (diversity
    mode); otherwise it is zero-filled.  ``data_mode`` selects the fill style
    ("random" or "infnan").

    To re-enable sparse support, inspect tv.layout here and dispatch to
    separate _build_sparse_coo / _build_sparse_csr helpers as needed.
    """
    import torch

    ndim    = _clamp_int(_eval_int(model, tv.ndim), 0, MAX_DIM)
    dtype_i = _simplify_dtype(_eval_int(model, tv.dtype, FLOAT32))  # simple dtypes
    # Merge positive-index and negative-index array slots.
    # Constraint files emit Select(sizes, IntVal(-k)) for t.size(-k); Z3 treats
    # that as a separate slot from sizes[ndim-k], so the positive slot may stay
    # at its default (0) even when the negative slot is constrained to non-zero.
    # When the positive slot is non-zero (explicitly constrained), use it directly
    # to prevent a free negative slot from overriding it via max().
    raw_pos  = [_eval_int(model, Select(tv.sizes,   IntVal(i)))         for i in range(ndim)]
    raw_neg  = [_eval_int(model, Select(tv.sizes,   IntVal(i - ndim)))  for i in range(ndim)]
    sizes    = _safe_sizes([p if p > 0 else max(p, n) for p, n in zip(raw_pos, raw_neg)])

    # Build a CONTIGUOUS tensor (standard row-major strides from `sizes`).
    #
    # The abstract environment folds is_contiguous → True and
    # is_non_overlapping_and_dense → True (abstract_env.cpp kTruePredicates), so
    # every constraint is derived under the assumption that inputs are contiguous.
    # The model therefore never constrains tv.strides; reading them back gives
    # unconstrained cells that default to all-1 strides — a non-contiguous,
    # overlapping layout that contradicts the model and breaks view/ravel-based
    # kernels (e.g. isin's `test_elements.ravel()` → "view size is not compatible
    # with input tensor's size and stride").  torch.empty(sizes) matches the
    # model's contiguity assumption.  (To exercise non-contiguous inputs we would
    # first have to teach the abstract env to reason about strides.)
    dtype = _torch_dtypes().get(dtype_i, torch.float32)
    try:
        t = torch.empty(sizes, dtype=dtype).zero_()
    except Exception:
        t = torch.zeros(sizes, dtype=dtype)

    if rng is not None:
        _randomize_(t, rng, data_mode)

    # Generic element-value clamp: honor any slice-declared [data_lo, data_hi]
    # bounds on this tensor (free bounds → None → no clamp).  Lets value-range
    # preconditions (e.g. normal's std >= 0) be expressed in the slice without any
    # op-specific knowledge here.  Guarded: clamp_ is undefined for bool/complex.
    lo = _eval_bound(model, tv.data_lo)
    hi = _eval_bound(model, tv.data_hi)
    if (lo is not None or hi is not None) and t.numel():
        import math
        if t.dtype == torch.bool:
            # clamp_ is undefined for bool; map the [lo, hi] window onto the
            # truth value.  lo >= 1 forces True (only True lies in [1, ...]);
            # hi <= 0 forces False.  Used e.g. by is_nonzero value-on-data
            # preconditions that require a scalar bool to be true.
            if lo is not None and lo > 0:
                t.fill_(True)
            elif hi is not None and hi < 1:
                t.fill_(False)
        elif t.is_complex():
            # clamp_ is undefined for complex; clamp the real and imaginary
            # parts independently so the bound still pushes magnitude away from 0.
            try:
                torch.view_as_real(t).clamp_(min=lo, max=hi)
            except Exception:
                pass
        else:
            # Integer tensors reject float clamp bounds ("Float can't be cast to
            # long"), so round into the dtype: ceil the lower bound, floor the
            # upper, keeping the clamp window within [lo, hi].  Needed e.g. for
            # scatter's int index whose values must lie in [0, self.size(dim)-1].
            if not t.is_floating_point():
                lo = math.ceil(lo) if lo is not None else None
                hi = math.floor(hi) if hi is not None else None
            try:
                t.clamp_(min=lo, max=hi)
            except Exception:
                pass

    requires_grad = _eval_bool(model, tv.requires_grad, default=False)
    if requires_grad and t.is_floating_point():
        t.requires_grad_(True)
    if _eval_bool(model, tv.is_conj, default=False) and t.is_complex():
        t = t.conj()
    if _eval_bool(model, tv.is_neg, default=False):
        t = t._neg_view()
    return t


def _build_scalar(sv: ScalarVar, model) -> Any:
    import math
    dtype_i = _eval_int(model, sv.dtype, INT64)
    if dtype_i == BOOL:
        return bool(_eval_int(model, sv.int_val, 0))
    if dtype_i in FLOATING_DTYPES:   # incl. float8 — a float8 scalar is a float
        if _eval_bool(model, sv.is_inf):
            return math.inf
        if _eval_bool(model, sv.is_nan):
            return math.nan
        return _eval_float(model, sv.real_val)
    if dtype_i in (COMPLEX32, COMPLEX64, COMPLEX128):
        if _eval_bool(model, sv.is_inf):
            return complex(math.inf, 0.0)
        if _eval_bool(model, sv.is_nan):
            return complex(math.nan, 0.0)
        return complex(_eval_float(model, sv.real_val), 0.0)
    return _eval_int(model, sv.int_val)


def _build_int_array(av: IntArrayVar, model) -> list[int]:
    length = _clamp_int(_eval_int(model, av.length), 0, MAX_INT_ARRAY_LEN)
    return [_eval_int(model, av[i]) for i in range(length)]


def _build_tensor_list(tl: TensorListVar, model, rng=None, data_mode: str = "random") -> list[Any]:
    length = _clamp_int(_eval_int(model, tl.length), 0, MAX_TENS)
    return [_build_tensor(tl.tensors[i], model, rng, data_mode)
            for i in range(length)]


def _build_scalar_list(sl: ScalarListVar, model) -> list[Any]:
    """Materialise the full ScalarVar pool as finite Python numbers.

    The list is sized to ``length`` here, but ``build_call_args`` resizes it to
    the sibling Tensor[] length afterwards (the foreach contract requires the two
    lists to be equal-length).  Complex / inf / nan scalar elements are coerced
    to a finite real so a Scalar[] never injects a value that fails for the wrong
    reason (e.g. a complex scalar into a real-tensor foreach op); restricting to
    finite reals is a sound subset of the valid Scalar[] inputs.
    """
    import math
    out: list[Any] = []
    for i in range(MAX_TENS):
        v = _build_scalar(sl.scalars[i], model)
        if isinstance(v, complex):
            v = v.real
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            v = 0.0
        out.append(v)
    # Return the full aligned pool; build_call_args truncates to the sibling
    # Tensor[] length so element i keeps its (constraint-pinned) pairing with
    # self[i].  Do NOT truncate by sl.length here — that would misalign and the
    # length-sync's tail-padding could reuse a wrong-typed scalar.
    return out


def _build_optional(typ: str, ov: OptVar, model, rng=None, data_mode: str = "random",
                    arg_name: str | None = None, op_name: str | None = None) -> Any:
    if not _eval_bool(model, ov.present):
        return None
    inner_type = typ[:-1]
    return _build_value(inner_type, ov.value, model, rng, data_mode, arg_name, op_name)


def _build_value(typ: str, var: Any, model, rng=None, data_mode: str = "random",
                 arg_name: str | None = None, op_name: str | None = None) -> Any:
    if typ.endswith("?") and isinstance(var, OptVar):
        return _build_optional(typ, var, model, rng, data_mode, arg_name, op_name)

    if typ == "Tensor":
        return _build_tensor(var, model, rng, data_mode)
    if typ == "Tensor[]":
        return _build_tensor_list(var, model, rng, data_mode)
    if typ == "Scalar[]":
        return _build_scalar_list(var, model)
    if typ == "int":
        return _eval_int(model, var)
    if typ == "int[]":
        return _build_int_array(var, model)
    if typ == "float":
        return _eval_float(model, var)
    if typ == "bool":
        return _eval_bool(model, var)
    if typ == "Scalar":
        return _build_scalar(var, model)
    if typ == "str":
        return _build_string_enum(_eval_int(model, var), arg_name, op_name)
    if typ == "Generator?" or isinstance(var, GeneratorStub):
        return None

    raise TypeError(f"No concretizer for argument type {typ!r}")
