"""
concretize.py — MOBILE-LOCAL COPY of z3/concretize.py, tweakable for the ExecuTorch
path independently of the main pipeline.

Converts a Z3 model into concrete PyTorch call arguments. The mobile pre-gen path injects
this copy into the generator via generate.opnode.set_build_call_args, so graphs for
the ExecuTorch fuzzer are concretized HERE — tweak dtype/shape selection, value
ranges, etc. (e.g. to bias toward what the portable runtime supports) without
touching the canonical z3/concretize.py.

Still shares model.py (the symbolic constraint model) with the main pipeline; only
the concretization logic is forked.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from z3 import BoolRef, IntNumRef, RatNumRef, Select, IntVal, is_true

from mobile.engine.model import (
    TensorVar, ScalarVar, IntArrayVar, OptVar, TensorListVar, ScalarListVar,
    GeneratorStub,
    MAX_DIM, MAX_TENS, MAX_INT_ARRAY_LEN,
    UINT8, INT8, INT16, INT32, INT64,
    FLOAT16, FLOAT32, FLOAT64,
    COMPLEX32, COMPLEX64, COMPLEX128,
    BOOL, BFLOAT16,
    FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E5M2FNUZ, FLOAT8_E4M3FNUZ, FLOAT8_E8M0FNU,
    UINT16, UINT32, UINT64,
    FLOATING_DTYPES,
)


@dataclass
class ConcreteArgs:
    args: list[Any] = field(default_factory=list)
    witness: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


_TORCH_DTYPE_MAP: dict[int, Any] | None = None

# Quantized ScalarType codes (c10::ScalarType gaps excluded from model.VALID_DTYPE_CODES).
# These appear only as a `dtype` *argument* of quantize ops (the op produces a
# quantized output); no quantized input *tensor* is ever built.  They are coerced
# to a torch.dtype ONLY for quantize ops (see _coerce_special_arg) so they never
# leak into other ops' optional `dtype` args.
QINT8, QUINT8, QINT32 = 12, 13, 14


def _torch_dtypes() -> dict[int, Any]:
    """Map model.py ScalarType int codes to torch.dtype objects.

    Quantized codes (QInt8/QUInt8/QInt32) are included so quantize ops' `dtype`
    argument materialises; _coerce_special_arg gates them to quantize ops only.
    """
    global _TORCH_DTYPE_MAP
    if _TORCH_DTYPE_MAP is None:
        import torch
        _TORCH_DTYPE_MAP = {
            QINT8:      torch.qint8,
            QUINT8:     torch.quint8,
            QINT32:     torch.qint32,
            UINT8:      torch.uint8,
            INT8:       torch.int8,
            INT16:      torch.int16,
            INT32:      torch.int32,
            INT64:      torch.int64,
            FLOAT16:    torch.float16,
            FLOAT32:    torch.float32,
            FLOAT64:    torch.float64,
            COMPLEX32:  torch.complex32,
            COMPLEX64:  torch.complex64,
            COMPLEX128: torch.complex128,
            BOOL:       torch.bool,
            BFLOAT16:   torch.bfloat16,
            FLOAT8_E5M2:     torch.float8_e5m2,
            FLOAT8_E4M3FN:   torch.float8_e4m3fn,
            FLOAT8_E5M2FNUZ: torch.float8_e5m2fnuz,
            FLOAT8_E4M3FNUZ: torch.float8_e4m3fnuz,
            FLOAT8_E8M0FNU:  torch.float8_e8m0fnu,
            UINT16: torch.uint16,
            UINT32: torch.uint32,
            UINT64: torch.uint64,
        }
    return _TORCH_DTYPE_MAP


# ── MOBILE dtype simplification ──────────────────────────────────────────────────
# Restrict generated leaf-tensor dtypes to a small, portable-runtime-friendly set:
#   float8, float16, float32, int, bool, simple complex.
# The solver may pick any dtype that satisfies the constraint; we collapse it to the
# representative of its kind. This runs post-solve and the emitted graph + eager
# oracle both use the concretized dtype, so the eager-vs-ExecuTorch diff stays
# consistent. It cuts the "0x12 unhandled-dtype" / exotic-dtype SKIPs.
#
# Tweak this set freely (e.g. drop float8 if randn-on-float8 SKIPs annoy you, or keep
# int32 distinct from int64). bool is kept because mask/condition leaves need it.
_SIMPLE_ALLOWED = {FLOAT8_E4M3FN, FLOAT16, FLOAT32, INT64, BOOL}
_SIMPLE_REMAP = {
    FLOAT64: FLOAT32, BFLOAT16: FLOAT16,
    FLOAT8_E5M2: FLOAT8_E4M3FN, FLOAT8_E5M2FNUZ: FLOAT8_E4M3FN,
    FLOAT8_E4M3FNUZ: FLOAT8_E4M3FN, FLOAT8_E8M0FNU: FLOAT8_E4M3FN,
    UINT8: INT64, INT8: INT64, INT16: INT64, INT32: INT64,
    UINT16: INT64, UINT32: INT64, UINT64: INT64,
    # complex dropped — ExecuTorch portable kernels are real-only, so complex
    # inputs mostly SKIP (isFloatingType / "Unhandled dtype ComplexFloat").
    COMPLEX32: FLOAT32, COMPLEX64: FLOAT32, COMPLEX128: FLOAT32,
}


def _simplify_dtype(code: int) -> int:
    """Collapse a model ScalarType code to the mobile simple-dtype set."""
    if code in _SIMPLE_ALLOWED:
        return code
    return _SIMPLE_REMAP.get(code, FLOAT32)   # anything exotic → float32


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
    dtype_i = _simplify_dtype(_eval_int(model, tv.dtype, FLOAT32))   # MOBILE: simple dtypes only
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
    return [_build_tensor(tl.tensors[i], model, rng, data_mode) for i in range(length)]


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


# Per-argument string enum maps (arg name → {int code: string}).  Used when a
# specific native argument only accepts a small known set of strings that the
# generic table below does not cover (e.g. gelu's `approximate` ∈ {none, tanh}).
# A slice/aigen constrains the arg's int var to this map's keys so every generated
# value materialises a string the operator actually accepts — letting the full
# valid space be reached instead of pinning to the single generic-table hit.
_STR_ENUM_BY_ARG = {
    "approximate": {0: "none", 1: "tanh"},   # gelu
    "UPLO": {0: "L", 1: "U"},                # linalg eigh/eigvalsh/cholesky/... (always L/U)
    "side": {0: "left", 1: "right"},         # searchsorted / bucketize
}

# Op-aware string-enum maps for args whose valid string set depends on the
# operator (arg-name alone collides).  Keyed by (op_base, arg_name) where
# op_base = op_name.split(".")[0].  Checked BEFORE the arg-name table.
#   - scatter.reduce / scatter.value_reduce use get_operator_enum(..., new=false)
#     → {add, multiply}.  (scatter_reduce.two / index_reduce use the new-options
#     {sum,prod,mean,amax,amin}; their aigens pin codes the generic table already
#     maps to valid strings, so they are deliberately NOT listed here.)
#   - linalg_qr `mode` ∈ {reduced, complete, r}.
_STR_ENUM_BY_OP_ARG = {
    ("scatter",   "reduce"): {0: "add", 1: "multiply"},
    ("linalg_qr", "mode"):   {0: "reduced", 1: "complete", 2: "r"},
    # pad mode ∈ {constant, reflect, replicate, circular} — NOT the interpolation
    # modes in the generic fallback (which caused "Unrecognised padding mode none").
    ("pad",       "mode"):    {0: "constant", 1: "reflect", 2: "replicate", 3: "circular"},
    # conv*d.padding / _convolution_mode `padding` string ∈ {valid, same}.
    ("conv1d",    "padding"): {0: "valid", 1: "same"},
    ("conv2d",    "padding"): {0: "valid", 1: "same"},
    ("conv3d",    "padding"): {0: "valid", 1: "same"},
    ("_convolution_mode", "padding"): {0: "valid", 1: "same"},
}


def _build_string_enum(value: int, arg_name: str | None = None,
                       op_name: str | None = None) -> str:
    # Op-aware first: (op_base, arg_name) overrides the generic arg-name table.
    if op_name is not None:
        op_base = op_name.split(".", 1)[0]
        per = _STR_ENUM_BY_OP_ARG.get((op_base, arg_name))
        if per is not None:
            return per.get(value, per[next(iter(per))])
    # Argument-aware: if the arg has a known valid string set, use it.
    per = _STR_ENUM_BY_ARG.get(arg_name)
    if per is not None:
        return per.get(value, per[next(iter(per))])
    # Generic fallback for unmapped string args.  This small stable table keeps
    # runtime verification moving; unsupported exact strings still become
    # inconclusive at the operator-call layer rather than at concretization.
    return {
        0: "none",
        1: "mean",
        2: "sum",
        3: "nearest",
        4: "linear",
        5: "bilinear",
        6: "bicubic",
    }.get(value, "none")


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


# Ops whose `dtype` argument selects a *quantized* output ScalarType.  Only these
# get QInt8/QUInt8/QInt32 coerced to a torch.dtype; for every other op a quantized
# code maps to None (default), preserving prior behaviour (q-codes never leak).
_QDTYPE_OP_PREFIXES = ("quantize_per_channel", "quantize_per_tensor")


def _coerce_special_arg(name: str, value: Any, op_name: str | None = None) -> Any:
    """Convert integer-valued special arguments to the PyTorch objects they represent."""
    if value is None:
        return None
    if name == "device":
        # CPU-only model: device is always None (use default CPU)
        return None
    if name in ("dtype", "dtype2") and isinstance(value, int):
        if value in (QINT8, QUINT8, QINT32):
            is_qop = op_name is not None and any(
                op_name.startswith(p) for p in _QDTYPE_OP_PREFIXES)
            if not is_qop:
                return None  # don't feed a quantized dtype to a non-quantize op
            return _torch_dtypes().get(value, None)            # keep q-dtype for q-ops
        # MOBILE: collapse a cast/factory `dtype=` arg (e.g. _to_copy/full/arange/
        # scalar_tensor) to the simple set, so it can't mint an int8/complex128/...
        # intermediate that a downstream kernel has no specialization for (0x12).
        return _torch_dtypes().get(_simplify_dtype(value), None)
    if name == "layout" and isinstance(value, int):
        import torch
        _LAYOUT_MAP = {0: torch.strided, 1: torch.sparse_coo}
        return _LAYOUT_MAP.get(value, torch.strided)
    if name == "memory_format" and isinstance(value, int):
        import torch
        _MEMFMT_MAP = {0: torch.contiguous_format, 1: torch.preserve_format,
                       2: torch.channels_last, 3: torch.channels_last_3d}
        return _MEMFMT_MAP.get(value, None)
    # rounding_mode (div.*_mode): valid values are None, 'trunc', 'floor' — the
    # generic string-enum produces 'none'/'mean'/… which are invalid here.
    if name == "rounding_mode":
        return None if value is None else "trunc"
    return value


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

    # Foreach Scalar[] length-sync: check_foreach_api_restrictions requires the
    # scalar list and the (first) tensor list to be the same length.  The model's
    # scalar-list length is free, so resize any built Scalar[] arg to the realised
    # Tensor[] length here (truncate; the pool is MAX_TENS so it is never short).
    tlist_len = None
    for (nm, ty), val in zip(named_params, result.args):
        base_ty = ty[:-1] if ty.endswith("?") else ty
        if base_ty == "Tensor[]" and isinstance(val, list):
            tlist_len = len(val)
            break
    if tlist_len is not None:
        for idx, (nm, ty) in enumerate(named_params):
            base_ty = ty[:-1] if ty.endswith("?") else ty
            if base_ty == "Scalar[]" and isinstance(result.args[idx], list):
                cur = result.args[idx]
                synced = (cur + [cur[-1] if cur else 0.0] * tlist_len)[:tlist_len]
                result.args[idx] = synced
                result.witness[nm] = synced

    return result
