"""Vulkan backend — GPU delegate. Lowers on the host; runs only on a device with Vulkan.
Partial-delegates: unsupported ops stay portable. No PT2E quantizer (fp16/fp32 path)."""

from __future__ import annotations

from .base import Backend, lower_with_partitioner


class VulkanBackend(Backend):
    name = "vulkan"
    runs_on_host = False
    supports_quantization = False

    def is_available(self) -> bool:
        try:
            from executorch.backends.vulkan.partitioner.vulkan_partitioner import (  # noqa: F401
                VulkanPartitioner,
            )
            return True
        except Exception:
            return False

    def _lower(self, ep, example_inputs):
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
            VulkanPartitioner,
        )
        return lower_with_partitioner(ep, VulkanPartitioner())
