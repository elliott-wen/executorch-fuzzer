"""
model.py — Symbolic type models for PyTorch argument types.

Every PyTorch native function argument is represented as a collection of
Z3 symbolic variables.  Higher-level Python classes group those variables
and expose computed properties (is_complex, numel, …) that appear as
branch conditions in the extracted LLVM IR paths.

Design decisions
────────────────
1. Flat Python classes, not Z3 algebraic datatypes.
   Z3 datatypes interact poorly with the array and arithmetic theories.
   Each property is a plain z3.Int / z3.Bool / z3.Array variable.

2. Z3 Array theory for sizes and strides (ArraySort(Int, Int)).
   Select(arr, symbolic_i) is a native, efficiently-solved operation.
   The alternative, SeqSort, is undecidable in general and too slow.

3. Bounded ndim (0 ≤ ndim ≤ MAX_DIM).
   Avoids unbounded quantifiers and makes numel() expressible as a
   finite product.  PyTorch practical paths rarely exceed 8 dimensions.

Only dense (strided) CPU tensors are modelled here.  Sparse, quantized,
MKL-DNN, and nested-tensor operators are out of scope.
"""

from __future__ import annotations

from typing import Any

from z3 import (
    Array, IntSort,
    Int, Real, Bool, BoolVal, IntVal,
    And, Or, Not, If, Implies, Select,
)


# ══════════════════════════════════════════════════════════════════════════════
# ScalarType constants
# Mirrors c10::ScalarType in PyTorch's aten/src/ATen/ScalarType.h.
# ══════════════════════════════════════════════════════════════════════════════

UINT8      = 0   # torch.uint8
INT8       = 1   # torch.int8
INT16      = 2   # torch.int16
INT32      = 3   # torch.int32
INT64      = 4   # torch.int64
FLOAT16    = 5   # torch.float16   (Half)
FLOAT32    = 6   # torch.float32   (Float)
FLOAT64    = 7   # torch.float64   (Double)
COMPLEX32  = 8   # torch.complex32  (ComplexHalf)
COMPLEX64  = 9   # torch.complex64  (ComplexFloat)
COMPLEX128 = 10  # torch.complex128 (ComplexDouble)
BOOL       = 11  # torch.bool
BFLOAT16   = 15  # torch.bfloat16
# Packed-bits types (18–22)
BITS1X8    = 18
BITS2X4    = 19
BITS4X2    = 20
BITS8      = 21
BITS16     = 22
# Float8 variants (23–26)
FLOAT8_E5M2     = 23
FLOAT8_E4M3FN   = 24
FLOAT8_E5M2FNUZ = 25
FLOAT8_E4M3FNUZ = 26
# Extended unsigned integers (27–29)
UINT16     = 27
UINT32     = 28
UINT64     = 29
# Additional float8 (44)
FLOAT8_E8M0FNU  = 44
DTYPE_MAX  = 44  # inclusive upper bound for domain constraints

# Dtype groups used in is_* predicates below.
COMPLEX_DTYPES   = (COMPLEX32, COMPLEX64, COMPLEX128)
INTEGER_DTYPES   = (UINT8, INT8, INT16, INT32, INT64, UINT16, UINT32, UINT64)
FLOAT8_DTYPES    = (FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E5M2FNUZ, FLOAT8_E4M3FNUZ, FLOAT8_E8M0FNU)
# c10::isFloatingType() returns true for the float8 types too, so they belong to
# the floating family (a float8 scalar materializes to a Python float / double).
FLOATING_DTYPES  = (FLOAT16, FLOAT32, FLOAT64, BFLOAT16) + FLOAT8_DTYPES

# Complete set of valid c10::ScalarType codes (excludes quantized gaps 12-14,16-17,
# and the unmapped gap 30-43). Used by axioms() to exclude spurious dtype values.
VALID_DTYPE_CODES = (
    UINT8, INT8, INT16, INT32, INT64,
    FLOAT16, FLOAT32, FLOAT64,
    COMPLEX32, COMPLEX64, COMPLEX128,
    BOOL, BFLOAT16,
    # BITS types (18-22) intentionally excluded: they are internal packed-bit
    # representation types never produced by user-facing PyTorch ops.  Including
    # them lets Z3 escape dtype-exclusion constraints (e.g. AT_DISPATCH paths
    # that forbid all normal dtypes) by picking a bits type, producing spurious
    # SAT witnesses that do not trigger real errors.
    FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E5M2FNUZ, FLOAT8_E4M3FNUZ,
    UINT16, UINT32, UINT64,
    FLOAT8_E8M0FNU,
)

# Human-readable name map used by the concretizer.
DTYPE_TO_TORCH: dict[int, str] = {
    UINT8: "torch.uint8",     INT8: "torch.int8",     INT16: "torch.int16",
    INT32: "torch.int32",     INT64: "torch.int64",   FLOAT16: "torch.float16",
    FLOAT32: "torch.float32", FLOAT64: "torch.float64",
    COMPLEX32: "torch.complex32", COMPLEX64: "torch.complex64",
    COMPLEX128: "torch.complex128", BOOL: "torch.bool",
    BFLOAT16: "torch.bfloat16",
}


# ══════════════════════════════════════════════════════════════════════════════
# DeviceType constants
# Mirrors c10::DeviceType (int8_t) in c10/core/DeviceType.h.
# ══════════════════════════════════════════════════════════════════════════════

DEVICE_CPU         =  0
DEVICE_CUDA        =  1
DEVICE_XPU         = 12
DEVICE_MPS         = 13
DEVICE_META        = 14
DEVICE_PRIVATEUSE1 = 20
DEVICE_TYPE_MAX    = 20


# ══════════════════════════════════════════════════════════════════════════════
# Layout constants
# Only LAYOUT_STRIDED is modelled (dense-only scope).
# To re-enable sparse, add LAYOUT_SPARSE_COO/CSR/CSC/BSR/BSC here,
# update LAYOUT_MAX, and lift the self.layout == LAYOUT_STRIDED axiom
# in TensorVar.axioms().
# ══════════════════════════════════════════════════════════════════════════════

LAYOUT_STRIDED = 0
LAYOUT_MAX     = 0


# ══════════════════════════════════════════════════════════════════════════════
# MemoryFormat constants
# Mirrors c10::MemoryFormat (int8_t) in c10/core/MemoryFormat.h.
# ══════════════════════════════════════════════════════════════════════════════

MEMORY_FORMAT_CONTIGUOUS       = 0
MEMORY_FORMAT_PRESERVE         = 1
MEMORY_FORMAT_CHANNELS_LAST    = 2
MEMORY_FORMAT_CHANNELS_LAST_3D = 3
MEMORY_FORMAT_MAX              = 3


# ══════════════════════════════════════════════════════════════════════════════
# QScheme constants
# Mirrors c10::QScheme (uint8_t) in c10/core/QScheme.h.
# ══════════════════════════════════════════════════════════════════════════════

QSCHEME_PER_TENSOR_AFFINE              = 0
QSCHEME_PER_CHANNEL_AFFINE             = 1
QSCHEME_PER_TENSOR_SYMMETRIC           = 2
QSCHEME_PER_CHANNEL_SYMMETRIC          = 3
QSCHEME_PER_CHANNEL_AFFINE_FLOAT_PARAMS = 4
QSCHEME_MAX                            = 4


# ══════════════════════════════════════════════════════════════════════════════
# Capacity bounds
# ══════════════════════════════════════════════════════════════════════════════

# Upper bound on tensor rank.  PyTorch supports up to 64, but practical
# TORCH_CHECK paths rarely branch on ndim > 8.  Larger values slow numel()
# because it unrolls into more multiplication terms for Z3.
MAX_DIM = 5

# Upper bound on the *length* of an int[] argument.  Deliberately DECOUPLED from
# MAX_DIM (tensor rank): some ops take a fixed int[] longer than the max rank —
# 3-D padding ops (reflection_pad3d / replication_pad3d and their backward/out
# variants) require `padding.length == 2*3 == 6` while tensor rank stays <= 5.
# Reduction `int[] dim` lists are still effectively capped at the tensor rank by
# their own per-element range + distinctness constraints, so this wider bound
# does not let them grow past ndim.
MAX_INT_ARRAY_LEN = 6

# Upper bound on the number of tensors in a TensorList argument.
MAX_TENS = 16


# ══════════════════════════════════════════════════════════════════════════════
# TensorVar
# ══════════════════════════════════════════════════════════════════════════════

class TensorVar:
    """
    Symbolic model of a single dense (strided) PyTorch tensor.

    Z3 variables
    ────────────
    ndim          : Int              rank, 0 ≤ ndim ≤ MAX_DIM
    sizes         : Array(Int→Int)   sizes[i] for i in [0, ndim); sizes[i] ≥ 0
    strides       : Array(Int→Int)   strides[i] for i in [0, ndim)
    dtype         : Int              c10::ScalarType value, 0 ≤ dtype ≤ DTYPE_MAX
    layout        : Int              always LAYOUT_STRIDED (== 0)
    device_type   : Int              always DEVICE_CPU (== 0)
    is_conj       : Bool             conjugate-view bit
    is_neg        : Bool             negation-view bit
    defined       : Bool             False iff the tensor has no storage
    requires_grad : Bool             True iff created with requires_grad=True
    is_nonzero    : Bool             True iff numel()==1 and the element is non-zero

    Computed properties (pure Z3 expressions, no new variables)
    ────────────────────────────────────────────────────────────
    dim(), size(i), stride(i), numel(), numel_is_one()
    is_complex(), is_floating_point(), is_integer(include_bool)
    is_float8(), is_reduced_floating()
    suggest_memory_format()
    """

    def __init__(self, name: str) -> None:
        self.name = name

        # Shape
        self.ndim    = Int  (f"{name}.ndim")
        self.sizes   = Array(f"{name}.sizes",   IntSort(), IntSort())
        self.strides = Array(f"{name}.strides", IntSort(), IntSort())

        # Scalar type and storage layout
        self.dtype  = Int(f"{name}.dtype")
        self.layout = Int(f"{name}.layout")

        # Device
        self.device_type = Int(f"{name}.device_type")

        # View metadata bits
        self.is_conj = Bool(f"{name}.is_conj")
        self.is_neg  = Bool(f"{name}.is_neg")

        self.requires_grad  = Bool(f"{name}.requires_grad")
        self.is_nonzero     = Bool(f"{name}.is_nonzero")

        # Opaque Int — compare against MEMORY_FORMAT_* constants.
        self._memory_format  = Int (f"{name}.memory_format")

        # Note: `defined` is False when the tensor has no storage (Tensor::defined()).
        self.defined         = Bool(f"{name}.defined")
        self.storage_offset  = Int (f"{name}.storage_offset")

        # Name IDs: 0 = unnamed (wildcard), positive = name ID (c10::Dimname).
        self._names          = Array(f"{name}.names", IntSort(), IntSort())

        self._is_subclass_like = Bool(f"{name}.is_subclass_like")
        self._is_zerotensor    = Bool(f"{name}.is_zerotensor")

        # Quantization metadata (free when not quantized)
        self.is_quantized = Bool(f"{name}.is_quantized")
        self.qscheme      = Int (f"{name}.qscheme")

        self.requires_grad = Bool(f"{name}.requires_grad")
        self.is_mkldnn     = Bool(f"{name}.is_mkldnn")

        # Auxiliary factorization witnesses: other_prod[i] = numel / sizes[i]
        # when dimension i is active and sizes[i] > 0.  Needed so Z3 can prove
        # numel % sizes[i] == 0 (InferSize / reshape reachability checks).
        self.other_prod = [Int(f"{name}.other_prod_{i}") for i in range(MAX_DIM)]

        # Element-value bounds (Real).  Deliberately LEFT FREE — they are NOT added
        # to axioms(), so they stay out of the model unless a slice constrains them.
        # The concretizer reads them with model_completion=False: a free bound yields
        # no value (no clamp; full random range preserved), while a slice-declared
        # bound (e.g. std.data_lo >= 0 for normal) makes the concretizer clamp the
        # generated data into [data_lo, data_hi].  This lets element-value
        # preconditions Z3 can't otherwise express be declared in the slice.
        self.data_lo = Real(f"{name}.data_lo")
        self.data_hi = Real(f"{name}.data_hi")

    # ── domain axioms ──────────────────────────────────────────────────────────

    def axioms(self) -> list:
        """
        Validity constraints that must hold for any real tensor.
        Add these to the solver once, unconditionally, for every TensorVar
        that represents a user-supplied argument.
        """
        constraints = [
            And(self.ndim >= 0, self.ndim <= MAX_DIM),
            Or(*(self.dtype == d for d in VALID_DTYPE_CODES)),
            self.layout == LAYOUT_STRIDED,   # dense-only; lift when re-adding sparse
            self.device_type == DEVICE_CPU,
            And(self._memory_format >= 0, self._memory_format <= MEMORY_FORMAT_MAX),
            self.storage_offset >= 0,
        ]
        # Every active dimension must have a non-negative size.
        for i in range(MAX_DIM):
            constraints.append(
                Implies(IntVal(i) < self.ndim, Select(self.sizes, IntVal(i)) >= 0)
            )
        # Strides for active dimensions must be ≥ 1 (stride=0 causes SIGFPE in
        # ATen kernels, and the concretizer clamps to max(1, stride)).  Bounding
        # to ≥ 1 here ensures that when Z3 solves for "stride is even", it picks
        # the smallest even value ≥ 1, which is 2 — preserved by the clamping.
        for i in range(MAX_DIM):
            constraints.append(
                Implies(IntVal(i) < self.ndim, Select(self.strides, IntVal(i)) >= 1)
            )
        # Numel divisibility: for each active dimension with positive size,
        # witness that numel factors through that dimension.  This lets Z3 prove
        # numel % sizes[i] == 0 without non-linear reasoning over the nested If.
        for i in range(MAX_DIM):
            constraints.append(
                Implies(
                    And(self.ndim > IntVal(i), Select(self.sizes, IntVal(i)) > IntVal(0)),
                    And(
                        self.numel() == Select(self.sizes, IntVal(i)) * self.other_prod[i],
                        self.other_prod[i] >= IntVal(1),
                    )
                )
            )
        constraints.append(Implies(self.is_conj, self.is_complex()))
        # Negative-index aliasing: sizes[-j] == sizes[ndim-j] for j in 1..MAX_DIM.
        # Z3 Array theory treats sizes[-1] and sizes[ndim-1] as independent slots.
        # Without this axiom, concretize.py's max(pos, neg) path can pick a large
        # unconstrained negative slot even when the positive slot is forced to 0 by
        # a broadcastability constraint, silently violating that constraint at runtime.
        for j in range(1, MAX_DIM + 1):
            constraints.append(Implies(
                IntVal(j) <= self.ndim,
                Select(self.sizes, IntVal(-j)) == Select(self.sizes, self.ndim - IntVal(j)),
            ))
        return constraints

    # ── shape queries ──────────────────────────────────────────────────────────

    def dim(self):
        """Number of dimensions (same as ndim)."""
        return self.ndim

    def size(self, i):
        """Size along dimension i. i may be a Python int or a Z3 Int.

        Negative i is normalized: size(-k) → sizes[ndim - k].
        This mirrors PyTorch's maybe_wrap_dim semantics so that constraints
        like size(-3) correctly access the third-to-last dimension, not an
        unconstrained array slot at key -3.
        """
        if isinstance(i, int):
            if i < 0:
                return Select(self.sizes, self.ndim + IntVal(i))
            return Select(self.sizes, IntVal(i))
        return Select(self.sizes, If(i >= 0, i, self.ndim + i))

    def stride(self, i):
        """Stride along dimension i. i may be a Python int or a Z3 Int."""
        if isinstance(i, int):
            if i < 0:
                return Select(self.strides, self.ndim + IntVal(i))
            return Select(self.strides, IntVal(i))
        return Select(self.strides, If(i >= 0, i, self.ndim + i))

    def numel(self):
        """
        Total number of elements: product of all active sizes.

        Non-linear arithmetic when ndim > 1; prefer numel_is_one() when the
        path only checks for scalar tensors.
        """
        result = IntVal(1)
        for k in range(MAX_DIM):
            result = If(self.ndim > k, result * Select(self.sizes, IntVal(k)), result)
        return result

    def numel_is_one(self):
        """
        Linear encoding of numel() == 1: every active dimension has size 1.
        Preferred over numel() == 1 to avoid non-linear arithmetic.
        """
        return And(*[
            Implies(IntVal(k) < self.ndim, Select(self.sizes, IntVal(k)) == 1)
            for k in range(MAX_DIM)
        ])

    # ── dtype predicates ───────────────────────────────────────────────────────

    def is_complex(self):
        return Or(*(self.dtype == d for d in COMPLEX_DTYPES))

    def is_floating_point(self):
        return Or(*(self.dtype == d for d in FLOATING_DTYPES))

    def is_integer(self, include_bool: bool = False):
        base = Or(*(self.dtype == d for d in INTEGER_DTYPES))
        return Or(base, self.dtype == BOOL) if include_bool else base

    def is_float8(self):
        return Or(*(self.dtype == d for d in FLOAT8_DTYPES))

    def is_reduced_floating(self):
        """dtype ∈ {Half, BFloat16, Float8_*} — mirrors c10::isReducedFloatingType."""
        return Or(self.dtype == FLOAT16, self.dtype == BFLOAT16, self.is_float8())

    # ── layout / memory format ─────────────────────────────────────────────────

    def suggest_memory_format(self):
        """Opaque c10::MemoryFormat Int. Compare against MEMORY_FORMAT_* constants."""
        return self._memory_format

    # ── named-dimension accessors ──────────────────────────────────────────────

    def has_names(self):
        """True iff any active dimension has a non-zero name ID (Tensor::has_names())."""
        return Or(*[
            And(IntVal(i) < self.ndim, Select(self._names, IntVal(i)) != IntVal(0))
            for i in range(MAX_DIM)
        ])

    def dim_name(self, i):
        """Name ID of dimension i (0 = unnamed). i may be a Python int or z3.Int."""
        return Select(self._names, IntVal(i) if isinstance(i, int) else i)

    @property
    def is_subclass_like(self):
        """True iff the tensor is a subclass-like object (at::isTensorSubclassLike())."""
        return self._is_subclass_like

    @property
    def is_zerotensor(self):
        """True iff the tensor is a zero tensor (Tensor::_is_zerotensor())."""
        return self._is_zerotensor

    def __repr__(self) -> str:
        return f"TensorVar({self.name!r})"


# ══════════════════════════════════════════════════════════════════════════════
# ScalarVar
# ══════════════════════════════════════════════════════════════════════════════

class ScalarVar:
    """
    Symbolic model of a c10::Scalar.

    A Scalar is a tagged union of {int64_t, double, c10::complex<double>, bool}.

    Z3 variables
    ────────────
    dtype    : Int    active variant, using ScalarType codes (INT64, FLOAT64, etc.)
    int_val  : Int    value when the scalar holds an integer
    real_val : Real   value when the scalar holds a finite float
    is_inf   : Bool   True iff the scalar holds ±∞
    is_nan   : Bool   True iff the scalar holds NaN
    """

    def __init__(self, name: str) -> None:
        self.name     = name
        self.dtype    = Int (f"{name}.dtype")
        self.int_val  = Int (f"{name}.int_val")
        self.real_val = Real(f"{name}.real_val")
        self.is_inf   = Bool(f"{name}.is_inf")
        self.is_nan   = Bool(f"{name}.is_nan")

    def axioms(self) -> list:
        return [
            Or(*(self.dtype == d for d in VALID_DTYPE_CODES)),
            Implies(self.is_inf, self.is_floating_point()),
            Implies(self.is_nan, self.is_floating_point()),
            Not(And(self.is_inf, self.is_nan)),
        ]

    def is_floating_point(self):
        return Or(*(self.dtype == d for d in FLOATING_DTYPES))

    def is_integral(self, include_bool: bool = False):
        base = Or(*(self.dtype == d for d in INTEGER_DTYPES))
        return Or(base, self.dtype == BOOL) if include_bool else base

    def is_complex(self):
        return Or(*(self.dtype == d for d in COMPLEX_DTYPES))

    def is_boolean(self):
        return self.dtype == BOOL

    # Aliases matching c10::Scalar predicate spelling (Scalar::isFloatingPoint /
    # isBoolean), used by check_addr_scalar-style constraints.
    def is_floating(self):
        return self.is_floating_point()

    def is_bool(self):
        return self.is_boolean()

    def isInf(self):
        return self.is_inf

    def isNan(self):
        return self.is_nan

    def scalar_type(self):
        """
        Canonical ScalarType — mirrors c10::detail::scalar_type():
          isFloatingPoint() → FLOAT64, isComplex() → COMPLEX128,
          isBoolean() → BOOL, else → INT64
        """
        return If(self.is_floating_point(), IntVal(FLOAT64),
               If(self.is_complex(),        IntVal(COMPLEX128),
               If(self.is_boolean(),        IntVal(BOOL),
                                            IntVal(INT64))))

    def __repr__(self) -> str:
        return f"ScalarVar({self.name!r})"


# ══════════════════════════════════════════════════════════════════════════════
# OptVar
# ══════════════════════════════════════════════════════════════════════════════

class OptVar:
    """
    Symbolic model of std::optional<T>.

    Z3 variables
    ────────────
    present : Bool    True iff the optional holds a value
    value   : T       the inner symbolic object (TensorVar, z3.Int, etc.)
    """

    def __init__(self, name: str, inner: Any) -> None:
        self.name    = name
        self.present = Bool(f"{name}.present")
        self.value   = inner
        self.val     = inner  # alias: X.val.ndim matches Z3 var name 'X.val.ndim'

    def has_value(self):
        return self.present

    def value_or(self, default):
        return If(self.present, self.value, default)

    def axioms(self) -> list:
        if hasattr(self.value, "axioms"):
            return self.value.axioms()
        return []

    def __repr__(self) -> str:
        return f"OptVar({self.name!r}, {self.value!r})"


# ══════════════════════════════════════════════════════════════════════════════
# IntArrayVar
# ══════════════════════════════════════════════════════════════════════════════

class IntArrayVar:
    """
    Symbolic model of c10::ArrayRef<int64_t> (IntArrayRef / int[]).

    Z3 variables
    ────────────
    length : Int              number of elements, 0 ≤ length ≤ MAX_DIM
    data   : Array(Int→Int)   data[i] for i in [0, length)
    """

    def __init__(self, name: str) -> None:
        self.name   = name
        self.length = Int  (f"{name}.length")
        self.data   = Array(f"{name}.data", IntSort(), IntSort())

    def axioms(self) -> list:
        # Only bound the length.  Elements are deliberately NOT forced >= 0: an
        # int[] may legitimately hold NEGATIVE values — dim indices (roll/sum/...
        # accept -ndim..ndim-1), the reshape/view "-1" infer sentinel, etc.  The
        # per-element value range is bounded by the harness's _add_size_axioms
        # ([-MAX_NDIM, MAX_SIZE]); ops that require non-negative sizes constrain
        # that themselves (their check-paths capture size>=0 / size>0).  Forcing
        # >= 0 here was a model-wide over-constraint that silently blocked valid
        # negative-element inputs for every int[] arg.
        return [And(self.length >= 0, self.length <= MAX_INT_ARRAY_LEN)]

    def __getitem__(self, i):
        """Element access — i may be a Python int or a Z3 Int."""
        return Select(self.data, IntVal(i) if isinstance(i, int) else i)

    def size(self):
        """Number of elements (mirrors ArrayRef::size())."""
        return self.length

    def __repr__(self) -> str:
        return f"IntArrayVar({self.name!r})"


# ══════════════════════════════════════════════════════════════════════════════
# TensorOptionsVar
# ══════════════════════════════════════════════════════════════════════════════

class TensorOptionsVar:
    """
    Symbolic model of c10::TensorOptions.

    Fields
    ──────
    dtype       : Int   ScalarType code
    layout      : Int   Layout code
    device_type : Int   always IntVal(0) (CPU-only model)
    """

    def __init__(self, name: str, dtype=None, layout=None) -> None:
        from z3 import Int, IntVal
        self.name        = name
        self.dtype       = dtype  if dtype  is not None else Int(f"{name}.dtype")
        self.layout      = layout if layout is not None else Int(f"{name}.layout")
        self.device_type = IntVal(0)  # CPU-only model

    def __repr__(self) -> str:
        return f"TensorOptionsVar({self.name!r})"


# ══════════════════════════════════════════════════════════════════════════════
# TensorListVar
# ══════════════════════════════════════════════════════════════════════════════

class TensorListVar:
    """
    Symbolic model of c10::ArrayRef<Tensor> (Tensor[] / TensorList).

    Z3 variables
    ────────────
    length  : Int                  number of tensors, 0 ≤ length ≤ MAX_TENS
    tensors : [TensorVar×MAX_TENS] fixed pool; tensors[i] valid when i < length
    """

    def __init__(self, name: str) -> None:
        self.name    = name
        self.length  = Int(f"{name}.length")
        self.tensors = [TensorVar(f"{name}[{i}]") for i in range(MAX_TENS)]

    def axioms(self) -> list:
        # Require at least 1 tensor: verification targets TORCH_CHECKs that fire
        # after the non-empty guard; an empty list takes the early-return path.
        constraints = [And(self.length >= 1, self.length <= MAX_TENS)]
        for t in self.tensors:
            constraints.extend(t.axioms())
        return constraints

    def __getitem__(self, i):
        """
        Access the i-th tensor. i may be a Python int or a Z3 Int.
        Symbolic indices produce a nested ITE chain over all MAX_TENS slots.
        """
        if isinstance(i, int):
            return self.tensors[i]
        result = self.tensors[MAX_TENS - 1]
        for k in range(MAX_TENS - 2, -1, -1):
            result = If(i == k, self.tensors[k], result)
        return result

    def __repr__(self) -> str:
        return f"TensorListVar({self.name!r}, len≤{MAX_TENS})"


# ══════════════════════════════════════════════════════════════════════════════
# ScalarListVar
# ══════════════════════════════════════════════════════════════════════════════

class ScalarListVar:
    """
    Symbolic model of c10::ArrayRef<Scalar> (Scalar[] / ScalarList) — e.g. the
    `scalars`/`exponent`/`weight` argument of the _foreach_*.ScalarList ops.

    Z3 variables
    ────────────
    length  : Int                  number of scalars, 1 ≤ length ≤ MAX_TENS
    scalars : [ScalarVar×MAX_TENS] fixed pool; scalars[i] valid when i < length

    The foreach contract (check_foreach_api_restrictions) requires the scalar
    list and the tensor list to have equal length; that length-match is enforced
    by the concretizer (it sizes the materialised list to the sibling Tensor[]),
    so a constraint need not tie `length` itself.
    """

    def __init__(self, name: str) -> None:
        self.name    = name
        self.length  = Int(f"{name}.length")
        self.scalars = [ScalarVar(f"{name}[{i}]") for i in range(MAX_TENS)]

    def axioms(self) -> list:
        constraints = [And(self.length >= 1, self.length <= MAX_TENS)]
        for s in self.scalars:
            constraints.extend(s.axioms())
        return constraints

    def __getitem__(self, i):
        """Element access — i may be a Python int or a Z3 Int."""
        if isinstance(i, int):
            return self.scalars[i]
        result = self.scalars[MAX_TENS - 1]
        for k in range(MAX_TENS - 2, -1, -1):
            result = If(i == k, self.scalars[k], result)
        return result

    def size(self):
        return self.length

    def __repr__(self) -> str:
        return f"ScalarListVar({self.name!r}, len≤{MAX_TENS})"


# ══════════════════════════════════════════════════════════════════════════════
# GeneratorStub
# ══════════════════════════════════════════════════════════════════════════════

class GeneratorStub:
    """
    Symbolic model of std::optional<at::Generator>.

    Most TORCH_CHECK paths ignore generator properties, but a few (e.g.
    randperm_out_cpu) check whether the optional is present and what device
    type the generator belongs to.

    Z3 variables
    ────────────
    present     : Bool   True iff the optional holds a generator
    device_type : Int    c10::DeviceType of the generator (when present)
    """

    def __init__(self, name: str) -> None:
        self.name        = name
        self.present     = Bool(f"{name}.present")
        self.device_type = Int(f"{name}.device_type")

    def has_value(self):
        return self.present

    def axioms(self) -> list:
        return [
            self.device_type == DEVICE_CPU,
        ]

    def __repr__(self) -> str:
        return f"GeneratorStub({self.name!r})"


# ══════════════════════════════════════════════════════════════════════════════
# ArgRegistry
# Maps normalised PyTorch type strings to factory functions.
# ══════════════════════════════════════════════════════════════════════════════

_ARG_BUILDERS: dict[str, Any] = {
    "Tensor":     lambda n: TensorVar(n),
    "Tensor?":    lambda n: OptVar(n, TensorVar(f"{n}.val")),
    "Tensor[]":   lambda n: TensorListVar(n),
    "int":        lambda n: Int(n),
    "int?":       lambda n: OptVar(n, Int(f"{n}.val")),
    "int[]":      lambda n: IntArrayVar(n),
    "int[]?":     lambda n: OptVar(n, IntArrayVar(f"{n}.val")),
    "float":      lambda n: Real(n),
    "float?":     lambda n: OptVar(n, Real(f"{n}.val")),
    "bool":       lambda n: Bool(n),
    "bool?":      lambda n: OptVar(n, Bool(f"{n}.val")),
    "Scalar":     lambda n: ScalarVar(n),
    "Scalar?":    lambda n: OptVar(n, ScalarVar(f"{n}.val")),
    "Scalar[]":   lambda n: ScalarListVar(n),
    "Scalar[]?":  lambda n: OptVar(n, ScalarListVar(f"{n}.val")),
    "Generator?": lambda n: GeneratorStub(n),
    # str/str? are mode strings (reduction, rounding, etc.); modelled as opaque ints.
    "str":        lambda n: Int(n),
    "str?":       lambda n: OptVar(n, Int(f"{n}.val")),
}


# ══════════════════════════════════════════════════════════════════════════════
# Cross-tensor helpers
# ══════════════════════════════════════════════════════════════════════════════

def is_same(a: TensorVar, b: TensorVar):
    """
    True iff a and b are the same tensor object (pointer equality).
    Returns a free Z3 Bool — symmetric: is_same(a,b) == is_same(b,a).
    """
    n1, n2 = sorted([a.name, b.name])
    # Use underscores so the Z3 variable name is a valid Python identifier.
    safe_n1 = "".join(c if c.isalnum() else "_" for c in n1)
    safe_n2 = "".join(c if c.isalnum() else "_" for c in n2)
    return Bool(f"is_same_{safe_n1}_{safe_n2}")


def same_shape(a: TensorVar, b: TensorVar):
    """
    True iff a and b have identical rank and the same size at every dimension.
    Encodes: a.ndim == b.ndim ∧ ∀ i < ndim: a.sizes[i] == b.sizes[i]
    """
    return And(
        a.ndim == b.ndim,
        *[Implies(IntVal(i) < a.ndim, a.size(i) == b.size(i))
          for i in range(MAX_DIM)]
    )


# Name ID 0 means unnamed (wildcard) — mirrors c10::Dimname::isWildcard().
NO_NAME = 0


def are_names_equal(a: TensorVar, b: TensorVar):
    """
    True iff a and b have the same rank and identical name IDs at every dimension.
    Encodes: a.ndim == b.ndim ∧ ∀ i < ndim: a.name(i) == b.name(i)
    Use when the IR calls Tensor::names().equals(other.names()) or are_names_equal().
    """
    return And(
        a.ndim == b.ndim,
        *[Implies(IntVal(i) < a.ndim, a.dim_name(i) == b.dim_name(i))
          for i in range(MAX_DIM)]
    )


def is_expandable_to(a: "TensorVar", b: "TensorVar"):
    """
    True iff a's shape is broadcast-compatible with b (a can expand to b).
    Encodes: a.ndim <= b.ndim ∧ ∀j < a.ndim: a.size_from_right(j) == 1
                                              OR a.size_from_right(j) == b.size_from_right(j)
    where size_from_right(j) = size(ndim - 1 - j).
    Mirrors c10::is_expandable_to / Tensor::is_expandable_to.
    """
    conds = [a.ndim <= b.ndim]
    for j in range(MAX_DIM):
        a_idx = a.ndim - IntVal(j) - IntVal(1)
        b_idx = b.ndim - IntVal(j) - IntVal(1)
        conds.append(Implies(
            IntVal(j) < a.ndim,
            Or(
                Select(a.sizes, a_idx) == IntVal(1),
                Select(a.sizes, a_idx) == Select(b.sizes, b_idx),
            )
        ))
    return And(*conds)


def broadcastable(a: "TensorVar", b: "TensorVar"):
    """
    True iff a and b have broadcast-compatible shapes (can be broadcast together).
    Encodes: ∀j < min(a.ndim, b.ndim): a.size_from_right(j) == 1
                                        OR b.size_from_right(j) == 1
                                        OR a.size_from_right(j) == b.size_from_right(j)
    Mirrors at::infer_size_dimvector — fires RuntimeError when this fails.
    """
    conds = []
    for j in range(MAX_DIM):
        a_has = IntVal(j) < a.ndim
        b_has = IntVal(j) < b.ndim
        a_sz = Select(a.sizes, a.ndim - IntVal(j) - IntVal(1))
        b_sz = Select(b.sizes, b.ndim - IntVal(j) - IntVal(1))
        conds.append(Implies(
            And(a_has, b_has),
            Or(a_sz == IntVal(1), b_sz == IntVal(1), a_sz == b_sz),
        ))
    return And(*conds)


def expandable_to(src: "TensorVar", dst: "TensorVar"):
    """
    True iff `src` can be broadcast (expanded) to `dst`'s shape WITHOUT changing
    dst — i.e. the broadcast result equals dst.shape.  Used for in-place / fixed-
    shape outputs (resize not allowed): every input must expand into the output.
    Requires src.ndim <= dst.ndim and, per trailing dim, src==1 or src==dst.
    """
    conds = [src.ndim <= dst.ndim]
    for j in range(MAX_DIM):
        src_has = IntVal(j) < src.ndim
        src_sz = Select(src.sizes, src.ndim - IntVal(j) - IntVal(1))
        dst_sz = Select(dst.sizes, dst.ndim - IntVal(j) - IntVal(1))
        # Only constrain trailing dims that src actually has.
        conds.append(Implies(src_has,
                             Or(src_sz == IntVal(1), src_sz == dst_sz)))
    return And(*conds)


def _sz_from_right_or1(t: "TensorVar", j: int):
    """Trailing-dim size (j=0 is the last dim), or 1 if t has no such dim —
    the per-dim value that participates in broadcasting (at::infer_size)."""
    return If(IntVal(j) < t.ndim,
              Select(t.sizes, t.ndim - IntVal(j) - IntVal(1)),
              IntVal(1))


def out_is_broadcast(out: "TensorVar", *ins: "TensorVar"):
    """True iff `out.shape` equals the broadcast of all `ins` shapes EXACTLY —
    the output shape of a pointwise op (binary `add`/`mul`/comparison, ternary
    `where`/`addcmul`, …). The structured `_out` resize is now a hard check, so
    `out` must match this shape (or be empty). Assumes the inputs are mutually
    broadcastable (so the per-dim max is just 'whichever isn't 1'). Encodes:
      out.ndim == max_i ins[i].ndim
      out.size_from_right(j) == max_i (ins[i].size_from_right(j) or 1)
    """
    max_nd = ins[0].ndim
    for t in ins[1:]:
        max_nd = If(t.ndim >= max_nd, t.ndim, max_nd)
    conds = [out.ndim == max_nd]
    for j in range(MAX_DIM):
        # Fold the per-dim broadcast: result r, next size s -> if r==1 take s else keep r.
        # Given the inputs are broadcastable (r==1 or s==1 or r==s), this is exact AND
        # 0-size-correct: broadcast(1,0)=0, broadcast(0,1)=0, broadcast(0,0)=0 (a 0 dim,
        # being != 1, is kept and never overwritten by a sibling 1).
        res = IntVal(1)
        for t in ins:
            tj = _sz_from_right_or1(t, j)
            res = If(res == IntVal(1), tj, res)
        conds.append(Implies(IntVal(j) < out.ndim,
                             Select(out.sizes, out.ndim - IntVal(j) - IntVal(1)) == res))
    return And(*conds)


def itemsize(dtype):
    """Byte size of a scalar of `dtype` (c10::elementSize / Tensor::element_size).

    Maps each ScalarType code to its element byte width; defaults to 1 for any
    unmodelled code so element_size() stays positive."""
    return If(dtype == IntVal(COMPLEX128), IntVal(16),
           If(Or(dtype == IntVal(INT64), dtype == IntVal(FLOAT64),
                 dtype == IntVal(COMPLEX64)), IntVal(8),
           If(Or(dtype == IntVal(INT32), dtype == IntVal(FLOAT32),
                 dtype == IntVal(COMPLEX32)), IntVal(4),
           If(Or(dtype == IntVal(INT16), dtype == IntVal(FLOAT16),
                 dtype == IntVal(BFLOAT16)), IntVal(2),
              IntVal(1)))))   # UInt8/Int8/Bool and any unmodelled code → 1


# Runtime-state constants used in throw conditions.
# All set to False: for standard fresh test inputs these conditions never fire.
# check_contiguous is True: fresh strided tensors are always contiguous.
check_contiguous       = BoolVal(True)
check_memory_format    = BoolVal(False)
check_resizable        = BoolVal(False)
check_scaling          = BoolVal(False)
check_overlap          = BoolVal(False)
check_internal_overlap = BoolVal(False)
check_name             = BoolVal(False)
check_deterministic         = BoolVal(False)
check_deterministic_fill    = BoolVal(False)
check_sdp_backend           = BoolVal(False)


# ══════════════════════════════════════════════════════════════════════════════
# maybe_wrap_dim
# Mirrors c10::maybe_wrap_dim(dim, dim_post_expr, wrap_scalar).
# ══════════════════════════════════════════════════════════════════════════════

def maybe_wrap_dim(dim, ndim, _allow_wrap_scalar: bool = False):
    """
    Maps a (possibly negative) dimension index to its canonical [0, ndim) form.
    Returns If(dim >= 0, dim, dim + ndim).
    """
    return If(dim >= 0, dim, dim + ndim)


# ══════════════════════════════════════════════════════════════════════════════
# Standalone dtype predicates
# Accept a raw z3.Int dtype value; useful after promote_types() or for
# checking raw int arguments.
# ══════════════════════════════════════════════════════════════════════════════

def is_floating_type(dtype):
    """dtype ∈ {Half, Float, Double, BFloat16} — mirrors c10::isFloatingType."""
    return Or(*(dtype == d for d in FLOATING_DTYPES))


def is_integral_type(dtype, include_bool: bool = False):
    """dtype is an integer ScalarType — mirrors c10::isIntegralType."""
    base = Or(*(dtype == d for d in INTEGER_DTYPES))
    return Or(base, dtype == BOOL) if include_bool else base


def is_complex_type(dtype):
    """dtype ∈ {ComplexHalf, ComplexFloat, ComplexDouble}."""
    return Or(*(dtype == d for d in COMPLEX_DTYPES))


def isBitsType(dtype):
    """dtype ∈ {BITS1X8, …, BITS16} — mirrors c10::isBitsType."""
    return Or(*(dtype == d for d in (BITS1X8, BITS2X4, BITS4X2, BITS8, BITS16)))


def is_reduced_floating_type(dtype):
    """dtype ∈ {Half, BFloat16, Float8_*} — mirrors c10::isReducedFloatingType.
    Standalone version for raw z3.Int dtype (e.g. result of common_dtype() or
    promote_types()); use TensorVar.is_reduced_floating() when operating on a tensor.
    """
    return Or(
        dtype == FLOAT16,
        dtype == BFLOAT16,
        Or(*(dtype == d for d in FLOAT8_DTYPES)),
    )


def toRealValueType(dtype):
    """
    Maps a complex dtype to its real (floating) counterpart — mirrors c10::toRealValueType().
    Non-complex dtypes are returned unchanged.
        COMPLEX32  → FLOAT16
        COMPLEX64  → FLOAT32
        COMPLEX128 → FLOAT64
    """
    return If(dtype == COMPLEX32, FLOAT16,
           If(dtype == COMPLEX64, FLOAT32,
           If(dtype == COMPLEX128, FLOAT64,
           dtype)))


def toComplexType(dtype):
    """
    Maps a floating dtype to its complex counterpart — mirrors c10::toComplexType().
    Non-floating dtypes are returned unchanged.
        FLOAT16  → COMPLEX32
        FLOAT32  → COMPLEX64
        FLOAT64  → COMPLEX128
    """
    return If(dtype == FLOAT16, COMPLEX32,
           If(dtype == FLOAT32, COMPLEX64,
           If(dtype == FLOAT64, COMPLEX128,
           dtype)))


def detail_scalar_type(dtype):
    """
    Canonical ScalarType — mirrors c10::detail::scalar_type():
      isFloatingType → FLOAT64, isComplexType → COMPLEX128, BOOL → BOOL, else → INT64
    """
    return If(is_floating_type(dtype), IntVal(FLOAT64),
           If(is_complex_type(dtype),  IntVal(COMPLEX128),
           If(dtype == BOOL,           IntVal(BOOL),
                                       IntVal(INT64))))


# ══════════════════════════════════════════════════════════════════════════════
# canCast
# Exact encoding of c10::canCast (c10/core/ScalarType.h).
# ══════════════════════════════════════════════════════════════════════════════

def canCast(from_dtype, to_dtype):
    """
    True iff from_dtype can be safely cast to to_dtype.
    Three disallowed cases; everything else is castable:
      1. complex  → non-complex
      2. floating → integral (excl. bool)
      3. non-bool → bool
    """
    return Not(Or(
        And(is_complex_type(from_dtype), Not(is_complex_type(to_dtype))),
        And(is_floating_type(from_dtype), is_integral_type(to_dtype, False)),
        And(from_dtype != IntVal(BOOL), to_dtype == IntVal(BOOL)),
    ))


# ══════════════════════════════════════════════════════════════════════════════
# result_type / common_dtype
# Mirrors at::result_type and TensorIterator::common_dtype.
# ══════════════════════════════════════════════════════════════════════════════

def result_type(d1, d2):
    """at::result_type for two raw ScalarType values. Delegates to promote_types."""
    return promote_types(d1, d2)


def common_dtype(*tensors):
    """
    TensorIterator common_dtype: reduces promote_types over all tensor dtypes.
    Example: common_dtype(arg0, arg1) == FLOAT32
    """
    if not tensors:
        raise ValueError("common_dtype requires at least one tensor")
    result = tensors[0].dtype
    for t in tensors[1:]:
        result = result_type(result, t.dtype)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# promote_types
# Mirrors c10::promoteTypes() from c10/core/ScalarType.cpp.
# ══════════════════════════════════════════════════════════════════════════════

# Ordered list matching the index2dtype array in ScalarType.cpp.
# Quantized, bits, and float8 types are excluded (they raise in promoteTypes).
_PROMOTE_INDEX2DTYPE = [
    UINT8, INT8, INT16, INT32, INT64,
    FLOAT16, FLOAT32, FLOAT64,
    COMPLEX32, COMPLEX64, COMPLEX128,
    BOOL, BFLOAT16,
]

# _promoteTypesLookup[i][j] → result dtype (ScalarType int value).
# Rows and columns follow _PROMOTE_INDEX2DTYPE order.
_PROMOTE_TABLE = [
    #  u8   i8  i16  i32  i64  f16  f32  f64  c32  c64 c128  b1  bf16
    [   0,   2,   2,   3,   4,   5,   6,   7,   8,   9,  10,   0,  15],  # uint8
    [   2,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,   1,  15],  # int8
    [   2,   2,   2,   3,   4,   5,   6,   7,   8,   9,  10,   2,  15],  # int16
    [   3,   3,   3,   3,   4,   5,   6,   7,   8,   9,  10,   3,  15],  # int32
    [   4,   4,   4,   4,   4,   5,   6,   7,   8,   9,  10,   4,  15],  # int64
    [   5,   5,   5,   5,   5,   5,   6,   7,   8,   9,  10,   5,   6],  # float16
    [   6,   6,   6,   6,   6,   6,   6,   7,   9,   9,  10,   6,   6],  # float32
    [   7,   7,   7,   7,   7,   7,   7,   7,  10,  10,  10,   7,   7],  # float64
    [   8,   8,   8,   8,   8,   8,   9,  10,   8,   9,  10,   8,   9],  # complex32
    [   9,   9,   9,   9,   9,   9,   9,  10,   9,   9,  10,   9,   9],  # complex64
    [  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  10],  # complex128
    [   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  15],  # bool
    [  15,  15,  15,  15,  15,   6,   6,   7,   9,   9,  10,  15,  15],  # bfloat16
]


def promote_types(a, b):
    """
    Z3 encoding of c10::promoteTypes(a, b).
    a, b: Z3 Int holding c10::ScalarType values.
    Returns a Z3 Int expression for the promoted ScalarType.
    """
    N = len(_PROMOTE_INDEX2DTYPE)

    def _row(i: int):
        result = IntVal(_PROMOTE_TABLE[i][N - 1])
        for j in range(N - 2, -1, -1):
            result = If(b == _PROMOTE_INDEX2DTYPE[j], IntVal(_PROMOTE_TABLE[i][j]), result)
        return result

    result = _row(N - 1)
    for i in range(N - 2, -1, -1):
        result = If(a == _PROMOTE_INDEX2DTYPE[i], _row(i), result)
    # promote_types(d, d) == d for all d, including Float8 and other types not in
    # the table above.  Apply this as an outer guard so any unlisted dtype self-promotes
    # correctly rather than falling through to the table's default arm.
    result = If(a == b, a, result)
    # c10::promoteTypes throws for cross-type promotion involving uint16/32/64 or
    # any Float8 variant.  Return -1 so constraints like "_v1 IN {float types}"
    # correctly exclude these combinations and Z3 never generates them.
    _restricted = Or(
        a == IntVal(UINT16),  a == IntVal(UINT32),  a == IntVal(UINT64),
        b == IntVal(UINT16),  b == IntVal(UINT32),  b == IntVal(UINT64),
        *(a == IntVal(d) for d in FLOAT8_DTYPES),
        *(b == IntVal(d) for d in FLOAT8_DTYPES),
    )
    return If(And(_restricted, a != b), IntVal(-1), result)


def compatible_dtypes(d1, d2) -> "BoolRef":
    """True iff c10::promoteTypes(d1, d2) would not throw at runtime.

    Matches torch.promote_types exactly for the restricted dtypes (verified by an
    exhaustive sweep):
      * Float8 variants promote with NOTHING but themselves.
      * UInt16/32/64 promote only with a floating type (Half/Float/Double/BFloat16)
        — NOT with int/bool/complex/Float8 or a *different* wide-unsigned type.
      * Self-promotion (d1 == d2) is always valid.
    Every other (standard) pair promotes fine.

    Used as an axiom between every pair of dtype variables so Z3 never proposes an
    input whose promotion would crash — without over-excluding the pairs torch
    actually accepts (e.g. uint16 x float, which the old "any-restricted => only
    self" rule wrongly forbade).
    """
    f8_involved = Or(*(d1 == IntVal(d) for d in FLOAT8_DTYPES),
                     *(d2 == IntVal(d) for d in FLOAT8_DTYPES))
    _wide_u = (UINT16, UINT32, UINT64)
    d1_uwide = Or(*(d1 == IntVal(d) for d in _wide_u))
    d2_uwide = Or(*(d2 == IntVal(d) for d in _wide_u))
    d1_float = Or(*(d1 == IntVal(d) for d in FLOATING_DTYPES))
    d2_float = Or(*(d2 == IntVal(d) for d in FLOATING_DTYPES))
    # a wide-unsigned operand is only promotable against a floating partner
    uwide_ok = And(Implies(d1_uwide, d2_float), Implies(d2_uwide, d1_float))
    return Or(d1 == d2, And(Not(f8_involved), uwide_ok))


def z3_div_floor(a, b):
    """Z3 integer division matching C++ div_rtn<T>: truncation toward zero.

    Z3's integer `/` operator truncates toward zero, identical to C++ signed
    integer division.  For non-negative `a` and positive `b` (the common case
    in convolution n_blocks computations) this equals Python floor division.

    Use whenever the IR contains a `div_rtn<int64_t>(numerator, denominator)`
    call, e.g. in im2col / col2im n_blocks calculations.

    Both `a` and `b` must be Z3 integer expressions (ArithRef).  The caller is
    responsible for ensuring `b != 0` via guard constraints.
    """
    return a / b
