"""webgpu — the ExecuTorch WebGPU delegate, sitting on top of the portable kernels.

Portable's rules, unchanged, and nothing else. The reasoning is vulkan.py's, and it applies
here MORE directly than anywhere else in this package: `WebGPUPartitioner` does not decide
anything itself. It constructs a `VulkanPartitioner` and forwards `partition()` and
`ops_to_not_decompose()` to it, so which nodes get delegated — and therefore which nodes fall
back to portable kernels — is Vulkan's answer verbatim.

That makes this a partial delegate with the same shape as vulkan: a lowered .pte is a mix,
the delegated nodes run WebGPU compute shaders and the rest run portable kernels, and those
fallback nodes need portable's contracts for exactly the reasons portable.py records.

Because partitioning is IDENTICAL to vulkan's, this target is deliberately a copy of
vulkan's rather than something derived independently. If the two ever need to diverge, that
divergence is evidence the wrapper stopped being a pass-through — check
`webgpu/partitioner/partitioner.py` before writing the rule.

How identical: the lowered .pte is BYTE-IDENTICAL to vulkan's and carries vulkan's delegate
id (upstream asserts this in `test_lowers_identically_with_vulkan_backend_id`, and the WebGPU
runtime registers as `Backend{"VulkanBackend", ...}`). Which runtime executes the blob is a
LINK-TIME choice. Measured here on 300 single-op graphs: zero differences from vulkan, in
every statistic including per-graph .pte size. See lower/backends/webgpu.py.

That is exactly why nothing WebGPU-specific belongs in this file YET. A target says how a
RUNTIME is narrower than eager — but no lowering run can distinguish the two runtimes, since
lowering cannot even see which one will execute. Only a wgpu-native run can justify a rule
here. Until then a lowering failure is a vulkan lowering failure, and usually a bug to REPORT
rather than a call to stop generating. See vulkan.py for the worked version of that argument.
"""

from __future__ import annotations

from typing import Any

from mobile.generator.targets import portable


def axioms(op_name: str, params: list[tuple[str, str]], variables: dict[str, Any]) -> list:
    """Portable's rules, unchanged — the delegate adds none of its own yet."""
    return portable.axioms(op_name, params, variables)
