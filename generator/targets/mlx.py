"""mlx — the Apple MLX delegate, sitting on top of the portable kernels.

Portable's rules, unchanged, and nothing else yet.

`MLXPartitioner` is node-level capability-based (`MLXOperatorSupport.is_node_supported`), so
this is a PARTIAL delegate: the nodes it declines stay as portable kernels in the same .pte.
Those fallback nodes need portable's contracts for the reasons portable.py records, which is
why this target starts by inheriting them.

There is a second reason to add nothing beyond that here, specific to this backend. MLX runs
only on Apple Silicon, and nothing in this repo can execute an MLX .pte — the lane is
lowering-only (see lower/backends/mlx.py). A target's job is to describe how a RUNTIME is
narrower than eager, and that claim can only be checked by running the runtime. Every rule
in this file would therefore be unfalsifiable here: it would narrow generation on a
hypothesis, and the graphs it dropped would be the evidence that the hypothesis was wrong.

So the discipline for this module is stricter than for the others: add a rule only when it
comes from MLX's own documented op contract or from a real Apple-Silicon run, never from a
lowering failure observed on this box. A partitioner declining an op is not a reason to stop
generating it — declining is the partitioner working, and the op simply falls back.
"""

from __future__ import annotations

from typing import Any

from mobile.generator.targets import portable


def axioms(op_name: str, params: list[tuple[str, str]], variables: dict[str, Any]) -> list:
    """Portable's rules, unchanged — the delegate adds none of its own yet."""
    return portable.axioms(op_name, params, variables)
