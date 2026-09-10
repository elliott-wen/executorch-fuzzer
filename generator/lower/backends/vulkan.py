"""vulkan — GPU delegate.

Lowers on the host; runs only on a device with Vulkan. Partial-delegates: unsupported ops
stay portable. Float fp32 by default, or int8 with --quantize — Vulkan's only quantization
scheme is WEIGHT-ONLY, symmetric per-output-channel 8-bit weights on `linear`, with
activations left float.

A caveat that matters for a fuzzer specifically: the weight-only fusion fires only when the
linear WEIGHT is a compile-time constant (a get_attr node); a placeholder weight aborts
Vulkan's FusePatternsPass. This generator feeds every tensor as a graph input, so a generated
`linear` will not fuse under --quantize unless the generator freezes its weight into a
constant.
"""

from __future__ import annotations

from mobile.generator.lower.backends.base import Backend, QuantMode, lower_with_partitioner


class VulkanBackend(Backend):
    name = "vulkan"
    runs_on_host = False
    quant = QuantMode.OPTIONAL        # fp32, or int8 weight-only (linear) with --quantize

    def is_available(self) -> bool:
        try:
            from executorch.backends.vulkan.partitioner.vulkan_partitioner import (  # noqa: F401
                VulkanPartitioner,
            )
            return True
        except Exception:
            return False

    def quantizer(self):
        # Weight-only int8: symmetric, per-output-channel weight scales, no activation
        # quantization (is_dynamic=False). Only `linear` is annotated; other ops stay float.
        from executorch.backends.vulkan.quantizer.vulkan_quantizer import (
            VulkanQuantizer,
            get_symmetric_quantization_config,
        )
        quantizer = VulkanQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(is_dynamic=False, weight_bits=8))
        return quantizer

    def _lower(self, ep, example_inputs):
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
            VulkanPartitioner,
        )
        return lower_with_partitioner(ep, VulkanPartitioner())
