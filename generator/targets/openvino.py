"""openvino — Intel's OpenVINO delegate (CPU here), on the portable floor.

Portable's rules, unchanged, and nothing else yet — the same starting point as the other
delegate targets: whatever the partitioner declines runs a portable kernel that still needs
portable's contracts.

Unlike coreml, samsung, cadence and ethos-u this one RUNS locally end to end — the backend is
host-executable and local_client/openvino_client drives it over the broker — so a run here
gives lowering yield AND the OK/MISMATCH/SKIP split, which is the more useful evidence.

ISOLATION MATTERS HERE. OpenVINO and QNN corrupt each other's heap when co-loaded in one
process (QNN's presence breaks OpenVINO's runtime import_model with a glibc "unaligned tcache
chunk" abort), so a run must set MOBILE_BACKENDS=openvino. That is not a performance hint; a
full backend sweep without it aborts mid-run.

MEASURED. A 10,000-graph single-op run lowered 7,863 (79.3%) and executed them: OK 6,281,
MISMATCH 495, CRASH 21, SKIP 1,066. Of the 1,994 to_edge failures, two dtypes were 77%:

    1,127  KeyError: torch.bfloat16
      405  KeyError: torch.int16

`axioms` below constrains those away. NOT constrained, and left visible on purpose:
  * 211  OpConversionFailure from the OpenVINO frontend — a conversion failing, not a dtype
         being unrepresentable.
  *  86  "Node aten_pixel_shuffle_default ... was not decomposed or delegated" — the SAME
         partitioner-contract violation vulkan.py records, reproducing on a second and
         unrelated backend. Constraining the dtype that triggers it would hide a real bug.
  * 1,049 of the 1,066 execution SKIPs are one cause (error 0x1 at method.cpp). One thing to
         look at, not a tail — and an execution concern, not a generation one.
"""

from __future__ import annotations

from typing import Any

from z3 import IntVal, Or

from mobile.generator.constraints.model import (
    BOOL, FLOAT16, FLOAT32, FLOAT64, INT8, INT32, INT64, UINT8,
)
from mobile.generator.targets import portable, tensors

#: The dtypes the ExecuTorch path can hand OpenVINO. Read off the INLINE `dtype_mapping` at
#: openvino/frontend/pytorch/torchdynamo/compile.py:112 — the dict that actually raises —
#: NOT OpenVINO's general pt_to_ov_type_map, which is wider and would have misled:
#:
#:     inline map : float32, float64, float16, int64, int32, uint8, int8, bool
#:     general map: all of the above PLUS bfloat16, int16, complex32/64/128, float8_*
#:
#: So bfloat16 and int16 ARE representable in OpenVINO and simply absent from the dict this
#: code path consults. That makes it arguably an OpenVINO bug rather than a capability limit,
#: and worth reporting upstream — but until it is fixed, generating them only wastes graphs.
OPENVINO_DTYPES = (FLOAT32, FLOAT64, FLOAT16, INT64, INT32, UINT8, INT8, BOOL)


def axioms(op_name: str, params: list[tuple[str, str]], variables: dict[str, Any]) -> list:
    """Portable's rules, plus the dtypes the ExecuTorch->OpenVINO path can map."""
    out = portable.axioms(op_name, params, variables)
    for var in variables.values():
        for tensor in tensors(var):
            out.append(Or(*(tensor.dtype == IntVal(d) for d in OPENVINO_DTYPES)))
    return out
