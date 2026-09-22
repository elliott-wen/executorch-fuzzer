"""portable — what the ExecuTorch portable CPU kernels require beyond eager.

No `allows` here on purpose: the portable kernel library is the floor every ExecuTorch
runtime falls back to, so it runs whatever the catalog offers and excludes nothing. A
DELEGATING target is where an op delta earns its keep — it takes a narrow set of ops and
leaves the rest to these kernels.

Every rule here was measured: each cites the check that refused it and how often, from a
10,000-graph single-operator corpus where a refusal names exactly one op and needs no
bisection. Eager accepts all of these calls; only this runtime does not.
"""

from __future__ import annotations

from typing import Any

from z3 import And, Implies, IntVal, Not, Or, Select

from mobile.generator.targets import tensors
from mobile.generator.constraints.model import (
    BFLOAT16,
    BOOL,
    COMPLEX32,
    COMPLEX64,
    COMPLEX128,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    OptVar,
    ScalarVar,
    TensorVar,
)

#: dtypes torch::executor::isFloatingType() accepts
FLOATING = (FLOAT16, FLOAT32, FLOAT64, BFLOAT16)

#: The portable kernels' common dispatch set, ET_SWITCH_REALHBBF16_TYPES: the real types plus
#: Bool, Half and BFloat16. Complex, the extended unsigned ints and every float8 are outside
#: it — 7,808 of 19,041 run-time refusals on a 100k corpus were "Unhandled dtype".
RUNNABLE_DTYPES = (UINT8, INT8, INT16, INT32, INT64,
                   FLOAT16, FLOAT32, FLOAT64, BOOL, BFLOAT16)

#: Ops whose kernel dispatches on a DIFFERENT set than the usual one. Overriding the dtype
#: set per op is how an exception is expressed — not by exempting the op from the rule, which
#: would leave it free to pick dtypes it equally cannot run.
#:
#: _conj_physical is the whole reason this exists. Nearly every portable kernel expands to
#: ET_SWITCH_REALHBBF16_TYPES; this one expands to ET_SWITCH_COMPLEXH_TYPES and implements
#: ONLY complex (op__conj_physical.cpp:35). Under the general rule it was fed reals and
#: refused every single time — 366 refusals on a 10k 8-op corpus, 36% of them, and the only
#: op in the catalog that never executed once.
DTYPES_BY_OP: dict[str, tuple[int, ...]] = {
    "_conj_physical": (COMPLEX32, COMPLEX64, COMPLEX128),
}

#: Arguments the kernels are stricter about than the dtype set alone. index_util.cpp asserts
#: `index.scalar_type() == ScalarType::Long`; eager takes int32 or int64, so the solver picks
#: int32 half the time — 267 of 818 refusals before this rule existed.
INDEX_ARGS = frozenset({"index"})


def _floating(var) -> list:
    """`var` must hold a floating dtype — nothing if the argument is absent.

    Rules take arguments one at a time rather than assuming a particular overload's
    signature. An overload that lacks one (mean.default has no `out`) must still get every
    OTHER clause: dropping a whole rule because one argument is missing is how
    `mean.default(x, dtype=complex64)` kept reaching the runtime.
    """
    if var is None:
        return []
    inner = var.value if isinstance(var, OptVar) else var
    dtype = inner.dtype if isinstance(inner, TensorVar) else inner
    rule = Or(*(dtype == IntVal(d) for d in FLOATING))
    return [Implies(var.present, rule)] if isinstance(var, OptVar) else [rule]


def axioms(op_name: str, params: list[tuple[str, str]], variables: dict[str, Any]) -> list:
    """Everything this runtime additionally requires of a call to `op_name`.

    The general dtype and argument rules, plus whichever per-op rule applies. A rule that
    cannot apply to this overload — an argument it expected is absent — contributes nothing
    rather than costing the whole op.
    """
    out = _general(op_name, variables)
    rule = PER_OP.get(op_name) or PER_OP.get(op_name.split(".", 1)[0])
    if rule is not None:
        out += list(rule(variables))
    return out


def _general(op_name: str, variables: dict[str, Any]) -> list:
    """Rules that hold for every op here, bar the per-op dtype overrides."""
    runnable = (DTYPES_BY_OP.get(op_name) or DTYPES_BY_OP.get(op_name.split(".", 1)[0])
                or RUNNABLE_DTYPES)
    out = []
    for name, var in variables.items():
        for tensor in tensors(var):
            if name in INDEX_ARGS:
                out.append(tensor.dtype == IntVal(INT64))
            else:
                out.append(Or(*(tensor.dtype == IntVal(d) for d in runnable)))
    return out


# ── per-op rules ────────────────────────────────────────────────────────────────────
# Each names the kernel check it satisfies.

def _native_group_norm(v: dict[str, Any]) -> list:
    """normalization_ops_util.cpp:160-161 — `in.size(0) == N` and `in.size(1) == C`.

    Eager only checks the PRODUCT (`X.numel() == N * C * HxW`), so it happily accepts a
    (2,4,3) input described as N=1,C=8,HxW=3; the portable kernel wants the dimensions
    themselves. 71 refusals, the single largest cause in the single-op corpus.
    """
    inp = v["input"]
    return [inp.ndim >= 2,
            Select(inp.sizes, IntVal(0)) == v["N"],
            Select(inp.sizes, IntVal(1)) == v["C"],
            v["group"] >= 1,
            v["C"] % v["group"] == 0]


def _convolution_backward(v: dict[str, Any]) -> list:
    """op_convolution_backward.cpp:41,43 — "Transposed Convolution not supported" and
    "Only 2D Convolution Backward supported". 43 refusals."""
    rules = []
    if v.get("transposed") is not None:
        rules.append(Not(v["transposed"]))
    if isinstance(v.get("weight"), TensorVar):
        rules.append(v["weight"].ndim == IntVal(4))
    return rules


def _fft_r2c(v: dict[str, Any]) -> list:
    """op_fft_r2c.cpp:32 — "onesided=False is not supported yet". 10 refusals."""
    return [v["onesided"]] if v.get("onesided") is not None else []


def _copy(v: dict[str, Any]) -> list:
    """op_copy.cpp:33 — `non_blocking == false`. 26 refusals here, 1,849 on the 100k
    multi-op corpus."""
    return [Not(v["non_blocking"])] if v.get("non_blocking") is not None else []


def _expand_copy(v: dict[str, Any]) -> list:
    """`implicit == false` in check_expand_copy_args. 26 refusals here, 1,885 at 100k."""
    return [Not(v["implicit"])] if v.get("implicit") is not None else []


def _pdist_forward(v: dict[str, Any]) -> list:
    """tensor_util.h:617 — the input must be rank 2, and op_pdist_forward.cpp:45 dispatches
    over ET_SWITCH_FLOATH_TYPES alone ("Unhandled dtype Int/Byte/Bool"). 37 + 10 refusals."""
    self_ = v.get("self")
    if not isinstance(self_, TensorVar):
        return []
    return [self_.ndim == IntVal(2)] + _floating(self_)


def _convolution(v: dict[str, Any]) -> list:
    """kernel_ops_util.cpp:376 — "Expect input tensor to be 3-D or 4-D", with `weight` and
    `out` then required to match that rank. Eager also does 3-D convolution, so a rank-5
    input is a runtime limit rather than an invalid call. 24 refusals."""
    inp = v.get("input")
    if not isinstance(inp, TensorVar):
        return []
    out = [Or(inp.ndim == IntVal(3), inp.ndim == IntVal(4))]
    weight = v.get("weight")
    if isinstance(weight, TensorVar):
        out.append(weight.ndim == inp.ndim)
    return out


#: The batch-norm companion tensors, which normalization_ops_util.cpp requires to carry the
#: input's dtype and to be sized like its channel dimension.
_BATCH_NORM_COMPANIONS = ("weight", "bias", "running_mean", "running_var")


def _batch_norm(v: dict[str, Any]) -> list:
    """check_batch_norm_args → tensors_have_same_dtype(in, weight/bias/running_*) at
    normalization_ops_util.cpp:33-39, and tensors_have_same_size_at_dims(·, 0, in, C_dim)
    at :55-60. Eager promotes a Half input against a Float running_mean; the kernel wants
    them already equal. 20 dtype + 4 size refusals."""
    inp = v.get("input")
    if not isinstance(inp, TensorVar):
        return []
    out = [inp.ndim >= 2]
    for name in _BATCH_NORM_COMPANIONS:
        for tensor in tensors(v.get(name)):
            out.append(tensor.dtype == inp.dtype)
            out.append(tensor.ndim == IntVal(1))
            out.append(Select(tensor.sizes, IntVal(0)) == Select(inp.sizes, IntVal(1)))
    return out


def _log_softmax(v: dict[str, Any]) -> list:
    """The OPTIMIZED log_softmax kernel dispatches on ScalarType::Float alone —
    `default: ET_KERNEL_CHECK(context, false, ...)` at op_log_softmax.cpp:152, with a
    standing `// TODO: support Double as well`. Eager takes fp16 and fp64 happily."""
    return [t.dtype == IntVal(FLOAT32)
            for t in (v.get("self"), v.get("out")) if isinstance(t, TensorVar)]


def _mean(v: dict[str, Any]) -> list:
    """check_mean_dim_args (reduce_util.cpp:370-386). With a `dtype` given it must be a
    floating type and `out` must match it; without one, BOTH input and output must be
    floating. Constraining all three to floating satisfies either branch. 161 refusals."""
    out = _floating(v.get("self")) + _floating(v.get("out")) + _floating(v.get("dtype"))
    dtype, result = v.get("dtype"), v.get("out")
    if isinstance(dtype, OptVar) and isinstance(result, TensorVar):
        out.append(Implies(dtype.present, result.dtype == dtype.value))
    return out


def _cumsum(v: dict[str, Any]) -> list:
    """tensor_util.h:411 — `dim >= -upper_bound && dim < upper_bound`, where upper_bound is
    the input's rank. The solver is free to pick a dim outside it. 72 refusals."""
    self_, dim = v.get("self"), v.get("dim")
    if not isinstance(self_, TensorVar) or dim is None:
        return []
    return [self_.ndim >= 1, dim >= -self_.ndim, dim < self_.ndim]


#: The widest integer range `extract_scalar` accepts whatever C type it targets: int8 stops
#: at 127 and uint8 rejects negatives, so [0, 127] is the intersection over every dispatch.
_SCALAR_MIN, _SCALAR_MAX = 0, 127


def _scalar_is_plain(scalar, low: int = _SCALAR_MIN) -> list:
    """`utils::extract_scalar` (scalar_utils.h:24) converts a c10::Scalar into the output's
    C type: it refuses a non-integral scalar outright, then refuses one outside that type's
    range. The out dtype is not known here, so the rule takes the range every type accepts.
    """
    if scalar is None:
        return []
    inner = scalar.value if isinstance(scalar, OptVar) else scalar
    if not isinstance(inner, ScalarVar):
        return []
    return [Not(inner.is_inf), Not(inner.is_nan), inner.dtype == IntVal(INT64),
            inner.int_val >= IntVal(low), inner.int_val <= IntVal(_SCALAR_MAX)]


def _add(v: dict[str, Any]) -> list:
    """op_add.cpp:136 `extract_scalar(alpha, &alpha_val)` (43 refusals), and :118
    `common_type == a_type && check_alpha_type(...)` (38).

    The second wants the promotion to be a no-op. Making both scalars integral holds it for
    an integral or floating `self` — an integral scalar never promotes a tensor — but not
    for a Bool one, where promote_type_with_scalar(Bool, Long) is Long.
    """
    out = _scalar_is_plain(v.get("alpha")) + _scalar_is_plain(v.get("other"))
    self_ = v.get("self")
    if isinstance(self_, TensorVar) and isinstance(v.get("other"), ScalarVar):
        out.append(self_.dtype != IntVal(BOOL))
    return out


def _arange(v: dict[str, Any]) -> list:
    """op_arange.cpp:55,59,63 — `extract_scalar` on `start`, `end` and `step`. `step` is
    additionally required to be non-zero, or the range never terminates. 47 + 22 refusals."""
    return (_scalar_is_plain(v.get("start")) + _scalar_is_plain(v.get("end"))
            + _scalar_is_plain(v.get("step"), low=1))


PER_OP = {
    "_log_softmax": _log_softmax,
    "mean": _mean,
    "cumsum": _cumsum,
    "add": _add,
    "sub": _add,                      # op_add_sub_impl.h — the same alpha extraction
    "arange": _arange,
    "convolution": _convolution,
    "_native_batch_norm_legit": _batch_norm,
    "_native_batch_norm_legit_no_training": _batch_norm,
    "native_batch_norm": _batch_norm,
    "native_group_norm": _native_group_norm,
    "convolution_backward": _convolution_backward,
    "_fft_r2c": _fft_r2c,
    "copy": _copy,
    "expand_copy": _expand_copy,
    "_pdist_forward": _pdist_forward,
}
