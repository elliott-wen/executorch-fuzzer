"""CoreML backend (Apple) — delegates to the Apple Neural Engine / GPU / CPU via CoreML.
Available only where coremltools imports (macOS). Runs on the Mac host. Supports PT2E
quantization via CoreMLQuantizer.

Apple APIs are validated on macOS; on Linux this backend simply isn't available (is_available
returns False) and is omitted from the registry.
"""

from __future__ import annotations

from .base import Backend, QuantMode, lower_with_partitioner


class CoreMLBackend(Backend):
    name = "coreml"
    runs_on_host = True               # CoreML executes on the Mac host
    quant = QuantMode.OPTIONAL        # float, or int8 with --quantize

    def is_available(self) -> bool:
        try:
            import coremltools  # noqa: F401
            from executorch.backends.apple.coreml.partition.coreml_partitioner import (  # noqa: F401
                CoreMLPartitioner,
            )
            return True
        except Exception:
            return False

    def quantizer(self):
        from coremltools.optimize.torch.quantization import (
            LinearQuantizerConfig,
            QuantizationScheme,
        )
        from executorch.backends.apple.coreml.quantizer import CoreMLQuantizer

        config = LinearQuantizerConfig.from_dict(
            {
                "global_config": {
                    "quantization_scheme": QuantizationScheme.symmetric,
                    "milestones": [0, 0, 10, 10],
                    "activation_dtype": "qint8",
                    "weight_dtype": "qint8",
                }
            }
        )
        return CoreMLQuantizer(config)

    def _lower(self, ep, example_inputs):
        from executorch.backends.apple.coreml.partition.coreml_partitioner import (
            CoreMLPartitioner,
        )
        return lower_with_partitioner(ep, CoreMLPartitioner())
