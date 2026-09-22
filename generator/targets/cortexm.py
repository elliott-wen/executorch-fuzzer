"""cortex-m (Arm) — int8 CMSIS-NN kernels on a Cortex-M CPU, on the portable floor.

Portable's rules, unchanged, and nothing else yet.

PASS-BASED, not a delegate — the same shape as cadence, and it matters for how this target is
measured. CortexMPassManager rewrites quantized aten ops into `cortex_m` custom ops in place
rather than capturing a subgraph, so `lower.job.delegation` reports (0, N, 0) and the
delegated/portable split that steered xnnpack, samsung and coreml is blank here. The number
to read instead is how many .pte actually CALL a cortex_m kernel; a graph can lower perfectly
and still be pure portable fallback.

QuantMode.ALWAYS. int8-only, so the backend quantizes regardless of the flag and the
comparison runs against `quantized_reference`, never the fp32 eager oracle. That also puts it
in reach of the PT2E observer gaps that dominated nxp and appeared on cadence and ethos-u —
"histogram_cpu not implemented for 'Byte'/'Char'/'Int'/'Long'" — which are a PyTorch-side
limitation surfacing through the quantizer, NOT something a target rule should hide.

Known going in, and worth checking against the run rather than assuming: an earlier campaign
found the corpus ~3.5x smaller than ethos-u's, and traced it to ReplaceQuantNodesPass being an
ALL-OR-NOTHING rewrite — one unsupported node loses the whole graph — rather than to crashes
or quantization failures. If the yield is low, look there before reaching for a dtype rule.

There is no host client: the .pte runs on a Cortex-M device or the Corstone FVP, so a run here
measures stages 1-2 only.
"""

from __future__ import annotations

from typing import Any

from mobile.generator.targets import portable


def axioms(op_name: str, params: list[tuple[str, str]], variables: dict[str, Any]) -> list:
    """Portable's rules, unchanged. Nothing Cortex-M-specific yet — see the module docstring."""
    return portable.axioms(op_name, params, variables)
