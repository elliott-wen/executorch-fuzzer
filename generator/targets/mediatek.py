"""mediatek — the MediaTek NeuroPilot (APU) delegate, on the portable floor.

Portable's rules, unchanged, and nothing else yet — the same starting point as the other
delegate targets.

RUNS IN A DIFFERENT INTERPRETER. The NeuroPilot SDK ships a cp310 wheel and pins protobuf<4,
so it cannot live in the main 3.12 venv: lowering workers must be started with
`--python .venv-mtk/bin/python`. The parent process is unaffected — it never imports torch —
so only the worker interpreter changes.

MEASURED. A 10,000-graph single-op run lowered 6,052 (61.1%) — far better than the ~3-4% an
earlier campaign saw, but still the second-worst of the ten backends. Of 3,832 to_edge
failures, the two dtypes below were 1,041, and `axioms` constrains them away.

NOT constrained, and the reason matters — it is the distinction vulkan.py records, that a
backend REFUSING a call is policy while a backend CRASHING is a finding:
  * 191  "Compile error: NIR[0]: ReshapeLayer | MDLA: Cannot support ..."
  * 184  "Compile error: NIR[0]: ElementWiseSubLayer | MDLA: Cannot support ..."
         Refusals, but per-OP and per-SHAPE rather than per-dtype, so they need their own
         measurement before any exclusion — the cadence run showed what guessing costs there.
  * 147  IndexError: list index out of range, and 289 bare AssertionError. Those are the
         converter FAILING, not refusing, and must stay generatable.
"""

from __future__ import annotations

from typing import Any

from z3 import IntVal, Or

from mobile.generator.constraints.model import (
    BOOL, FLOAT32, FLOAT64, INT8, INT16, INT32, INT64, UINT8,
)
from mobile.generator.targets import portable, tensors

#: Portable's RUNNABLE_DTYPES minus bfloat16 and float16. Derived EMPIRICALLY, because the
#: NeuroPilot converter is a compiled Cython module (mtk_converter/.../importer_v2.py) whose
#: map cannot be read: attributing every lowering outcome in a 10,000-graph run to its leaf
#: dtypes isolates the two that raise KeyError inside _get_meta_val_type —
#:
#:     bfloat16   599 KeyError of 1,376 graphs carrying it
#:     float16    410 KeyError of   870
#:     every other dtype: 6-34, and those are graphs that ALSO carry a bf16/fp16 leaf
#:
#: float16 being absent is the surprising one for an NPU target, and it is what an earlier
#: campaign saw as "float16 compile errors" — same cause, seen one stage later.
MEDIATEK_DTYPES = (UINT8, INT8, INT16, INT32, INT64, FLOAT32, FLOAT64, BOOL)


def axioms(op_name: str, params: list[tuple[str, str]], variables: dict[str, Any]) -> list:
    """Portable's rules, plus the dtypes the NeuroPilot converter can map."""
    out = portable.axioms(op_name, params, variables)
    for var in variables.values():
        for tensor in tensors(var):
            out.append(Or(*(tensor.dtype == IntVal(d) for d in MEDIATEK_DTYPES)))
    return out
