"""vgf — Arm's VGF delegate (TOSA → ML SDK model-converter → Vulkan), on the portable floor.

Portable's rules, unchanged, and nothing else yet — the same starting point as vulkan.py and
for the same reason: VGF is a PARTIAL delegate, so a lowered .pte is a mix and every node the
partitioner declines runs a portable kernel that still needs portable's contracts.

Where VGF differs from the other targets is WHERE it loses graphs. Measured on 10,000
single-op graphs (--nodes 1 --target portable), lowered with --backend vgf and run on
lavapipe + the ML emulation layer:

    stage           vgf              vulkan (same corpus shape)
    lowered         8,163  82.3%     9,663  97.5%
    to_edge failed  1,729  17.4%       224   2.3%
    then executed   OK 94.6%, MISMATCH 1.6%, CRASH 0.2%, SKIP 3.6%

So VGF executes CLEANLY once a graph survives lowering — 13 crashes against Vulkan's 65, and
none of Vulkan's vkapi::Error abort storm. Its losses are almost entirely at to_edge, and
84% of those are dtype policy rather than bugs. That is what a target module can fix.

── WHAT A RULE COULD RECOVER (1,729 to_edge failures, categorised) ──────────────────────

  623  36%  INCONSISTENT DTYPE — every tensor of a node must share one dtype.
            e.g. log1p.out(int32 in, out=float64) -> "aten.log.default: Expected all tensors
            to have dtype DType.INT32, but found inconsistent dtype DType.FP64". Eager allows
            a mixed out=; TOSA does not. The message names the FIRST tensor's dtype as the
            expected one, which is why it can read as "sin expecting BOOL".
            RULE: all tensors in a call carry one dtype. Same shape as xnnpack's
            "does not support mixed input dtypes", but covering `out` as well.

  401  23%  DTYPE OUTSIDE TOSA'S SET — the accepted set is roughly
            {INT32, FP16, FP32, BF16}; bool, int64, uint8 and fp64 are all outside it.
            e.g. eq.Scalar_out(bool, -1, out=bool).
            RULE: restrict tensor dtypes to that set. Note this is NARROWER than portable's
            RUNNABLE_DTYPES and will exclude ops with a mandatory int64/bool slot, exactly as
            it did for xnnpack — expect an `allows` list to fall out of it.

  ~190 11%  UINT8 AT AN INTERNAL NODE — "Found internal uint8 tensor at node
            aten_neg_default. Uint8 is only allowed at IO boundaries." (~44% of the 429
            failures that ExecuTorch's pass manager buries under 120 repetitions of
            '_ExportedProgramGraphPassAdapter' with no cause attached; found by unwrapping
            __cause__.) A uint8 LEAF is legal — it stops being legal the moment decomposition
            puts a uint8 tensor inside the graph.
            RULE: the boundary-only distinction is not expressible in our per-call model,
            which sees one op and not where its tensors sit in the DAG. Dropping uint8
            entirely would cover it, at the cost of never testing VGF's uint8 IO path.

   51   3%  COMPLEX — unsupported; portable already refuses complex, so these arrive via ops
            portable pins to complex (DTYPES_BY_OP). An `allows` exclusion is the fit.

── WHAT A RULE MUST *NOT* SWALLOW ───────────────────────────────────────────────────────

Some to_edge failures are the Arm passes CRASHING, not refusing, and those are findings:

   24      ZeroDivisionError inside a pass
    ~4     IndexError: tuple index out of range
    ~3     ValueError: not enough values to unpack (expected 4, got 3)
    ~3     "No input quantization parameter found in node aten_bmm_default"

A dtype rule would stop generating the inputs that reach them, quietly closing a real bug
class. Whatever is added here should be checked against these counts afterwards: if they go
to zero, the rule went too far.

Nothing is added yet because the numbers above come from ONE run on CPU-emulated Vulkan, and
because the first two rules interact — constraining dtypes to the TOSA set may make the
consistency rule redundant, or may make whole ops unsatisfiable. Measure one at a time.
"""

from __future__ import annotations

from typing import Any

from mobile.generator.targets import portable


def axioms(op_name: str, params: list[tuple[str, str]], variables: dict[str, Any]) -> list:
    """Portable's rules, unchanged. See the module docstring for what VGF additionally wants
    and why none of it is written yet."""
    return portable.axioms(op_name, params, variables)
