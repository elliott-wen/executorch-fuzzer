"""Dtype tables for concretization.

Maps model.py ScalarType int codes to torch.dtype objects, and back. Nothing here decides
WHICH dtypes a graph may use; the solver picks, and this only translates the answer.
"""

from __future__ import annotations

from typing import Any

from mobile.generator.constraints.model import (
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


_CODE_BY_DTYPE: dict[Any, int] | None = None


def dtype_code_for(torch_dtype) -> int | None:
    """torch.dtype → its model ScalarType code, or None if the model has no code for it.

    The inverse of _torch_dtypes(), built once on first use. Needed wherever a CONCRETE
    tensor has to be described back to the solver — pinning a consumer's port to the
    dtype of the producer feeding it.
    """
    global _CODE_BY_DTYPE
    if _CODE_BY_DTYPE is None:
        _CODE_BY_DTYPE = {dt: code for code, dt in _torch_dtypes().items()}
    return _CODE_BY_DTYPE.get(torch_dtype)
