"""XNNPACK backend — CPU delegate. Runs on the host, so its corpus is directly testable
here (unlike vulkan/qnn/coreml). Partial-delegates: unsupported ops stay portable.
Supports PT2E quantization via XNNPACKQuantizer (symmetric int8)."""

from __future__ import annotations

from .base import Backend, lower_with_partitioner


class XnnpackBackend(Backend):
    name = "xnnpack"
    runs_on_host = True
    supports_quantization = True

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
        q = XNNPACKQuantizer()
        q.set_global(get_symmetric_quantization_config())
        return q

    def _lower(self, ep, example_inputs):
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        return lower_with_partitioner(ep, XnnpackPartitioner())
