"""samsung — the Exynos ENN delegate, on the portable floor.

Portable's rules, unchanged, and nothing else yet — the same starting point as vulkan.py,
vgf.py and qualcomm.py, and for the same reason: ENN is a PARTIAL delegate, so every node its
partitioner declines runs a portable kernel that still needs portable's contracts.

NO LOCAL EXECUTION. Unlike the other targets there is no host client for ENN — it needs a
real Exynos device, so a run here measures stages 1-2 (generate, lower) only. Delegation and
lowering yield are observable; OK/MISMATCH/SKIP are not.

── ONE RULE IS ALREADY KNOWN, FROM THE SOURCE RATHER THAN A RUN ────────────────────────

executorch/backends/samsung/builders/utils.py DATA_TYPE_STR_MAPPING is the complete set of
dtypes ENN can name:

    int8, uint8, int16, uint16, int32, int64, float16, float32

Portable's RUNNABLE_DTYPES is that set PLUS bool, float64 and bfloat16. A tensor of those
three has no ENN representation at all, so any node carrying one cannot be delegated — and
`bool` is the expensive one, since comparison and logical ops produce it constantly.

That prediction was then measured, on 10,000 single-op graphs lowered with --backend samsung.
Of 2,829 to_executorch failures — 28.5% of the corpus — 2,395 (85%) were exactly those three
dtypes, refused by name:

     960  RuntimeError: ('Data type cannot be decided: ', torch.bfloat16)
     926  RuntimeError: ('Data type cannot be decided: ', torch.bool)
     509  RuntimeError: ('Data type cannot be decided: ', torch.float64)

`axioms` below therefore restricts every tensor to DATA_TYPE_STR_MAPPING's keys. Unlike
vulkan.py — where a dtype rule would have HIDDEN partitioner bugs rather than fixed anything —
these are the backend naming a dtype it cannot represent, which is policy, not a defect.

WHAT THE RULE COSTS, and why `allows` is longer than the UNSAT set. Solving finds only six
ops with no valid call (complex, plus the two with a mandatorily bool `mask`). The comparison
and logical families are NOT among them — nothing in their extracted precondition forces a
bool result, so the solver happily picks another dtype for the declared output and the call
is satisfiable. But they still RETURN bool at run time, because that is their semantics rather
than a dtype variable, so they reached the backend and failed there instead: 881 bool refusals
survived the dtype rule, 878 of them from 11 families. They are excluded by name because a
constraint cannot reach them.

That costs nothing real. ENN has no bool representation at all, so those ops could never have
been delegated or even lowered — excluding them drops only the repeated confirmation that they
fail, and it stops ~11% of the corpus being spent on a known answer.

NOT COVERED, deliberately. A tail of ~430 failures is ENN gaps rather than dtype policy:
93 "aten.clone.default is not supported in ENN Delegate", 45 the same for aten.addmm.default,
46 Conv1dToConv2d pass failures, 61 bare AssertionError. Those stay generatable — they are
what this backend has left to tell us. If they drop to zero after a change here, the change
went too far.
"""

from __future__ import annotations

from typing import Any

from z3 import IntVal, Or

from mobile.generator.constraints.model import (
    FLOAT16, FLOAT32, INT8, INT16, INT32, INT64, UINT8, UINT16,
)
from mobile.generator.targets import excluding, portable, tensors

#: executorch/backends/samsung/builders/utils.py DATA_TYPE_STR_MAPPING — the complete set of
#: dtypes ENN can name. Portable's RUNNABLE_DTYPES additionally allows bool, float64 and
#: bfloat16, which have no ENN representation at all; UINT16 is here and not in portable, so
#: the intersection the solver sees is the seven types both accept.
ENN_DTYPES = (INT8, UINT8, INT16, UINT16, INT32, INT64, FLOAT16, FLOAT32)

#: The only ops with no satisfiable call left under ENN_DTYPES. Derived by solving every op
#: once with the axioms below applied and keeping the UNSAT ones — re-derive the same way
#: after changing ENN_DTYPES. Six of 221, in two groups:
#:   complex-typed, which ENN cannot name at all (portable PINS the first to complex via
#:   DTYPES_BY_OP, and the other two have no valid call under portable either);
#:   and the masked_* pair, whose `mask` argument is mandatorily bool.
#: Named rather than left to collapse: an op whose constraints have no model still LOADS and
#: then silently never generates, which reads as coverage.
allows = excluding(
    "_conj_physical", "_fft_c2r", "view_as_real_copy",   # complex
    "masked_fill", "masked_scatter",                     # bool mask
    # Ops whose RESULT is bool whatever the solver picks for the declared output dtype.
    # ENN_DTYPES cannot stop these: the axioms constrain the tensors the model describes,
    # and a comparison produces bool from its semantics, not from a dtype variable. Measured:
    # these 11 families caused 878 of the 881 remaining "Data type cannot be decided:
    # torch.bool" failures on a 10,000-graph run WITH the dtype rule already applied.
    "eq", "ne", "lt", "le", "gt", "ge",                  # comparisons
    "logical_and", "logical_or", "logical_xor", "logical_not",
    "any",                                               # bool reduction
)


def axioms(op_name: str, params: list[tuple[str, str]], variables: dict[str, Any]) -> list:
    """Portable's rules, plus the dtypes ENN is able to name."""
    out = portable.axioms(op_name, params, variables)
    for var in variables.values():
        for tensor in tensors(var):
            out.append(Or(*(tensor.dtype == IntVal(d) for d in ENN_DTYPES)))
    return out
