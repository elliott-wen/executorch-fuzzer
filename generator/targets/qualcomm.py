"""qualcomm — the QNN delegate (Qualcomm AI Engine Direct), on the portable floor.

Portable's rules, unchanged, and nothing else yet — the same starting point as vulkan.py and
vgf.py, and for the same reason: QNN is a PARTIAL delegate, so a lowered .pte is a mix and
every node the partitioner declines runs a portable kernel that still needs portable's
contracts.

Nothing measured here yet. The other three targets were written from a 10,000-graph
single-op run each, and the interesting thing was that they wanted OPPOSITE treatment:

    vgf      82.3% lowered, 17.4% to_edge failures, 84% of them DTYPE POLICY
             -> rules would recover most of it
    vulkan   97.5% lowered,  2.3% to_edge failures, ~70% of them ExecuTorch failing
             INTERNALLY (partitioner reserves pixel_shuffle then declines it;
             FuseBatchNormPass asserting on a graph with no batch norm)
             -> rules would HIDE the bugs, so none were added
    xnnpack  delegation, not run rate, was the number that moved: 3.2% -> 12.4% of ops
             once tensors were constrained to the fp32/fp16 envelope its partitioner takes

So the first question for QNN is which of those three it resembles, and the evidence is the
same three numbers: the to_edge failure breakdown, `lower.job.delegation`, and the per-op
skip/mismatch table from a --nodes 1 corpus (one op per graph, so every failure names its own
op and needs no bisection).

One thing known in advance, from the backend module rather than a run: QNN's HTP is an int8
and fp16 device. QuantMode is OPTIONAL, so the float path targets fp16 — which makes a dtype
rule a likely candidate, in the xnnpack mould. It is deliberately NOT written yet, because
whether it raises delegation or merely hides partitioner bugs is exactly what the run is for.
"""

from __future__ import annotations

from typing import Any

from mobile.generator.targets import portable


def axioms(op_name: str, params: list[tuple[str, str]], variables: dict[str, Any]) -> list:
    """Portable's rules, unchanged. Nothing QNN-specific yet — see the module docstring."""
    return portable.axioms(op_name, params, variables)
