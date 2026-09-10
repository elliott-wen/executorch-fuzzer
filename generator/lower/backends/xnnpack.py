"""xnnpack — CPU delegate.

Runs on the host, so its corpus is directly testable here (unlike vulkan/qnn/coreml).
Partial-delegates: ops XNNPACK does not support stay portable. Supports PT2E quantization
via XNNPACKQuantizer (symmetric int8).
"""

from __future__ import annotations

from mobile.generator.lower.backends.base import Backend, QuantMode, lower_with_partitioner


class XnnpackBackend(Backend):
    name = "xnnpack"
    runs_on_host = True
    quant = QuantMode.OPTIONAL        # float fp32, or int8 with --quantize

    def is_available(self) -> bool:
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import (  # noqa: F401
                XnnpackPartitioner,
            )
            return True
        except Exception:
            return False

    def quantizer(self):
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config())
        return quantizer

    def _lower(self, ep, example_inputs):
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        return lower_with_partitioner(ep, XnnpackPartitioner())
