"""cadence — the Cadence Xtensa DSP backend, on the portable floor.

Portable's rules, unchanged, and nothing else yet — the same starting point as the other
delegate targets, and for the same reason: whatever Cadence's passes decline runs a portable
kernel that still needs portable's contracts.

TWO THINGS MAKE THIS TARGET UNLIKE THE OTHERS, and both should shape whatever is added here.

  1. QuantMode.ALWAYS. Cadence is int8-only: the backend quantizes regardless of the
     --quantize flag, so the graph the DSP sees is a PT2E-quantized one and the comparison
     runs against `quantized_reference`, not the fp32 eager oracle. A dtype rule written as
     if the float dtypes reached the backend would be aimed at the wrong graph.

  2. It is PASS-BASED, not a partitioner/delegate. Cadence rewrites ops into cadence:: kernels
     in place rather than capturing a subgraph, so `lower.job.delegation` reports (0, N, 0)
     for it — the delegated/portable split that steered xnnpack, vulkan, vgf and samsung is
     blank here. The number that matters instead is how many .pte actually CALL a cadence::
     kernel; a graph can lower perfectly and still be pure portable fallback.

Both of those numbers now exist, from a 10,000-graph single-op run (8,863 lowered, 89.4%),
executed locally on the host-CPU reference kernels (local_client/cadence_client --target
generic): OK 8,572 / MISMATCH 117 / CRASH 2 / SKIP 172, in 12.6 s. That comparison is
BIT-EXACT rather than tolerance-based — the generic kernels ARE the reference semantics of the
cadence:: ops — so a nonzero delta is a genuine AoT-lowering or reference-kernel bug.

`allows` below therefore excludes only what provably cannot lower. No dtype axiom is added:
unlike samsung, where three named dtypes caused 85% of the losses, cadence's failures are
per-OP (ops absent from the Core ATEN opset), so naming the ops is both the precise fix and
the honest one.

WHAT IS LEFT, AND MUST STAY. These are findings, not noise, and should not be constrained
away — if they disappear after a change here, the change went too far:
  * native_group_norm 32/32, addmm.out 15/15, _native_batch_norm_legit.no_stats 17/17,
    remainder.Tensor_out 7/7, prod.int_out 7/7 — all MISMATCH against a bit-exact reference.
  * deltas of 1.082e9 and 255.0 on an int8 quantized path: range/overflow, not rounding.
  * 204 quant_ref failures, "histogram_cpu not implemented for 'Byte'" — the PT2E observer
    cannot handle a uint8 input. A PyTorch gap surfacing through Cadence's quantizer.
"""

from __future__ import annotations

from typing import Any

from mobile.generator.targets import excluding, portable

#: Ops Cadence cannot lower AT ALL, measured on 10,000 single-op graphs: each failed 100% of
#: its attempts at to_executorch, with the reason naming the op. Excluded because the answer
#: is already known and re-deriving it costs ~4% of every corpus.
#:
#: NOT the whole bitwise family — only the SHIFTS. bitwise_and/or/xor/not lower cleanly here
#: (148/155/134/49 attempts, 100% ready), so excluding "bitwise" wholesale would have thrown
#: away working coverage. The shifts alone are absent from the Core ATen opset.
allows = excluding(
    # SpecViolationError: "... is not in Core ATen opset"
    "bitwise_left_shift",    # 137 attempts, 137 failures
    "bitwise_right_shift",   # 139 attempts, 139 failures
    "_conj_physical",        #  59 attempts,  59 failures — complex-only
    "_fft_r2c",              #  55 attempts,  55 failures — produces complex
    # AttributeError: 'complex' object has no attribute 'data', raised inside the pass that
    # rewrites the subtraction rsub decomposes into.
    "rsub",                  #  36 failures
)


def axioms(op_name: str, params: list[tuple[str, str]], variables: dict[str, Any]) -> list:
    """Portable's rules, unchanged. Nothing Cadence-specific yet — see the module docstring."""
    return portable.axioms(op_name, params, variables)
