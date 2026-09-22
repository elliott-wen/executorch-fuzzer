"""coreml — Apple's Core ML delegate, on the portable floor.

Portable's rules, unchanged, and nothing else yet — the same starting point as the other
delegate targets, and for the same reason: whatever the partitioner declines runs a portable
kernel that still needs portable's contracts.

LOWERS HERE, RUNS ELSEWHERE. coremltools does the AoT conversion on Linux, so stages 1-2
measure normally; but a .mlpackage can only EXECUTE on Apple hardware, so the differential
half needs a Mac worker attached to the broker. A run on this box therefore reports lowering
yield, the failure breakdown and delegation — not OK/MISMATCH/SKIP.

MEASURED, then acted on. A 10,000-graph single-op run lowered only 4,084 (41.2%) — the worst
of any backend — and 5,806 died at to_edge. Three dtypes accounted for 3,708 of those (64%):

    1,680  KeyError: torch.bfloat16
    1,343  KeyError: torch.uint8
      685  KeyError: torch.int8

A BARE KeyError, not a validated rejection: coremltools has no entry for these at all, so
there is no graceful path and no diagnostic — which is why `axioms` below constrains them
away. This is the samsung case, not the vulkan one: the backend naming a dtype it cannot
represent is policy, and no finding is lost by not generating it.

NOT CONSTRAINED, deliberately:
  * 740  "To use fp16 input, please set minimum deployment target to iOS16+". That is a
         CONFIGURATION error, not a missing dtype — the fix belongs in
         lower/backends/coreml.py, and fp16 is Core ML's native ANE precision, so it is the
         last thing we should stop generating.
  * 215  "Deploying a model with no inputs in CoreML requires ..." — CoreML refusing a graph
         whose inputs are all baked constants. A real property of OUR generator, worth
         knowing rather than hiding.
  *  56  bitwise_and "only supports boolean input" — per-op, and the cadence run showed why
         that matters: excluding a whole bitwise family there would have discarded 486
         working graphs. Any exclusion here gets its own measurement first.

The five targets written before this one wanted three different treatments, and CoreML turned
out to be the samsung shape:

    vgf / samsung   dtype POLICY dominated the losses (84% / 85%) -> a dtype rule recovered
                    most of it: samsung went 70.2% -> 89.9% lowered
    cadence         per-OP gaps (ops absent from the Core ATen opset) -> naming the ops was
                    the precise fix, 89.4% -> 93.3%
    vulkan          ~70% ExecuTorch failing INTERNALLY -> a rule would have HIDDEN the bugs,
                    so none was added
    ethos-u         the QUANTIZER breaking (int64->int32 cast passes), not the backend

Two things to watch for specifically, since they would not show up as dtype policy:
  * Core ML is fp16-first on ANE and fp32 on CPU/GPU, with a compute_units choice behind it.
    A precision story like QNN's is plausible, and QNN's mismatches turned out NOT to be fp16
    rounding — so check the magnitudes before believing that explanation.
  * Earlier campaigns against CoreML on a Mac found a graph-optimisation DROPPED-STORE bug on
    aliased outputs. That is a MISMATCH class, not a lowering one, so it will only appear once
    a Mac worker is attached — and it is exactly the kind of finding a dtype rule must not be
    allowed to hide.
"""

from __future__ import annotations

from typing import Any

from z3 import IntVal, Or

from mobile.generator.constraints.model import (
    BOOL, FLOAT16, FLOAT32, FLOAT64, INT16, INT32, INT64,
)
from mobile.generator.targets import portable, tensors

#: coremltools TORCH_DTYPE_TO_MIL_DTYPE — the map consulted at
#: mil/frontend/torch/exir_utils.py:60, which is a plain dict lookup with no fallback. Read
#: off the installed coremltools rather than transcribed from docs:
#:
#:     accepts : bool, float16, float32, float64, int16, int32, int64
#:     missing : uint8, int8, bfloat16   -> bare KeyError, not a validated rejection
#:
#: Portable's RUNNABLE_DTYPES supplies all three of the missing ones, so the intersection the
#: solver sees drops exactly them. int16 is accepted here but immediately cast to int32 with a
#: warning ("Core ML does not support int16 input"), which is a conversion rather than a
#: refusal, so it stays.
COREML_DTYPES = (BOOL, FLOAT16, FLOAT32, FLOAT64, INT16, INT32, INT64)


def axioms(op_name: str, params: list[tuple[str, str]], variables: dict[str, Any]) -> list:
    """Portable's rules, plus the dtypes coremltools is able to map."""
    out = portable.axioms(op_name, params, variables)
    for var in variables.values():
        for tensor in tensors(var):
            out.append(Or(*(tensor.dtype == IntVal(d) for d in COREML_DTYPES)))
    return out
