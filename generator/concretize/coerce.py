"""Special-argument coercion: turn integer-valued args into PyTorch objects.

Handles `device`/`dtype`/`layout`/`memory_format`/`rounding_mode` and gates the
quantized dtype codes to quantize ops only (so q-dtypes never leak elsewhere).
"""

from __future__ import annotations

from typing import Any

from .dtypes import _torch_dtypes, QINT8, QUINT8, QINT32


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
        return _torch_dtypes().get(value, None)
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
