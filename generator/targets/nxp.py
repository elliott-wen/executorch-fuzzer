"""nxp — the eIQ Neutron NPU (i.MX RT700), on the portable floor.

Portable's rules, unchanged, and nothing else yet — the same starting point as the other
delegate targets: whatever the Neutron partitioner declines runs a portable kernel that still
needs portable's contracts.

QuantMode.ALWAYS. Neutron is an integer NPU, so the backend quantizes regardless of the
--quantize flag and the comparison runs against `quantized_reference`, not the fp32 eager
oracle — the same shape as cadence and ethos-u.

RUNS LOCALLY, bit-exactly. There is no ExecuTorch host runtime for a Neutron command stream
and no RT700 board here, but local_client/nxp_client drives the eIQ NSYS simulator — a
bit-exact C model of the NPU — so a nonzero diff is a real finding rather than a tolerance
question. That puts nxp in the small group (with cadence and openvino) where the differential
half is actually reachable on this machine.

Nothing measured yet. Two things are known going in and should be checked against the run
rather than assumed:
  * An earlier campaign here yielded only 0.5% until a missing quantized OUT-VARIANT
    registration was fixed, after which it reached 23.5%. If the yield comes back near the
    low number, suspect registration before writing any rule.
  * The residue after that fix was attributed to two NeutronQuantizer bugs. Bugs are exactly
    what a dtype rule must not hide — the distinction vulkan.py records.
"""

from __future__ import annotations

from typing import Any

from mobile.generator.targets import portable


def axioms(op_name: str, params: list[tuple[str, str]], variables: dict[str, Any]) -> list:
    """Portable's rules, unchanged. Nothing Neutron-specific yet — see the module docstring."""
    return portable.axioms(op_name, params, variables)
