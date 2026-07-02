"""Dtype tables for concretization.

Maps model.py ScalarType int codes to torch.dtype objects, and collapses the
solver's chosen dtype to a small, portable-runtime-friendly representative set.
See _simplify_dtype.
"""

from __future__ import annotations

from typing import Any

from mobile.gen.z3engine.model import (
    UINT8, INT8, INT16, INT32, INT64,
    FLOAT16, FLOAT32, FLOAT64,
    COMPLEX32, COMPLEX64, COMPLEX128,
    BOOL, BFLOAT16,
    FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E5M2FNUZ, FLOAT8_E4M3FNUZ, FLOAT8_E8M0FNU,
    UINT16, UINT32, UINT64,
)


_TORCH_DTYPE_MAP: dict[int, Any] | None = None

# Quantized ScalarType codes (c10::ScalarType gaps excluded from model.VALID_DTYPE_CODES).
# These appear only as a `dtype` *argument* of quantize ops (the op produces a
# quantized output); no quantized input *tensor* is ever built.  They are coerced
# to a torch.dtype ONLY for quantize ops (see coerce._coerce_special_arg) so they
# never leak into other ops' optional `dtype` args.
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
# Restrict generated leaf-tensor dtypes to a small, portable-runtime-friendly set.
# The solver may pick any dtype that satisfies the constraint; we collapse it to the
# representative of its kind. This runs post-solve and the emitted graph + eager
# oracle both use the concretized dtype, so the eager-vs-ExecuTorch diff stays
# consistent. It cuts the "0x12 unhandled-dtype" / exotic-dtype SKIPs.
#
# A code in _ALLOWED passes through unchanged; otherwise _REMAP collapses it:
#   - unsigned wide ints → their signed-equivalent width (uint16→int16, uint32→int32,
#     uint64→int64); signed ints and uint8 pass through as-is.
#   - exotic floats (float8/bf16) → a single representative float; float64 → float32.
#   - complex → float32 (ExecuTorch portable kernels are real-only).
# Anything unlisted falls back to float32. bool is kept (mask/condition leaves need it).
_ALLOWED = frozenset({FLOAT8_E4M3FN, FLOAT16, FLOAT32,
                      UINT8, INT8, INT16, INT32, INT64, BOOL})

_REMAP = {
    FLOAT64: FLOAT32, BFLOAT16: FLOAT16,
    FLOAT8_E5M2: FLOAT8_E4M3FN, FLOAT8_E5M2FNUZ: FLOAT8_E4M3FN, FLOAT8_E4M3FNUZ: FLOAT8_E4M3FN,
    FLOAT8_E8M0FNU: FLOAT8_E4M3FN,
    UINT16: INT16, UINT32: INT32, UINT64: INT64,
    COMPLEX32: FLOAT32, COMPLEX64: FLOAT32, COMPLEX128: FLOAT32,
}


def _simplify_dtype(code: int) -> int:
    """Collapse a model ScalarType code to the portable-runtime simple-dtype set."""
    if code in _ALLOWED:
        return code
    return _REMAP.get(code, FLOAT32)   # anything exotic → float32
