"""ethos-u — Arm's Ethos-U NPU (executed on the Corstone FVP), on the portable floor.

Portable's rules, unchanged, and nothing else yet — the same starting point as the other
delegate targets, and for the same reason: whatever the partitioner declines runs a portable
kernel that still needs portable's contracts.

QuantMode.ALWAYS, like cadence. Ethos-U is an INTEGER-ONLY NPU: an unquantized graph simply
cannot target it, so the backend quantizes regardless of the --quantize flag and the
comparison runs against `quantized_reference` rather than the fp32 eager oracle. Any rule
added here has to be about the graph the NPU actually sees.

Lowering goes through TOSA, as vgf does — Ethos-U is TOSA-INT where VGF is TOSA-FP+INT. That
makes vgf.py the closest reference for what to expect: its 1,729 to_edge failures were 84%
dtype policy, and the same dtype family is likely to appear here in the integer half.

Execution is the Corstone FVP simulator via local_client/fvp_client, which is slower per job
than the other host runners, so measure lowering first and size the execution run to it.
"""

from __future__ import annotations

from typing import Any

from mobile.generator.targets import portable


def axioms(op_name: str, params: list[tuple[str, str]], variables: dict[str, Any]) -> list:
    """Portable's rules, unchanged. Nothing Ethos-U-specific yet — see the module docstring."""
    return portable.axioms(op_name, params, variables)
