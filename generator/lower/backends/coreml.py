"""coreml (Apple) — delegates to the Neural Engine / GPU / CPU via CoreML.

coremltools converts on Linux, so the AoT half runs anywhere; EXECUTING a .mlpackage needs
Apple hardware. Supports PT2E quantization via CoreMLQuantizer.

The partitioner is given explicit compile specs rather than defaults, for one measured
reason — see MINIMUM_DEPLOYMENT_TARGET below.
"""

from __future__ import annotations

from mobile.generator.lower.backends.base import Backend, QuantMode, lower_with_partitioner


#: Deployment target handed to the partitioner. The DEFAULT is older than iOS16 and refuses a
#: float16 input outright — "To use fp16 input, please set minimum deployment target to
#: iOS16+" was 1,206 of 3,765 to_edge failures (32%) on a 10,000-graph single-op run, the
#: single largest cause once the missing-dtype KeyErrors were constrained away.
#:
#: That is a CONFIGURATION limit, not something the backend cannot do: fp16 is Core ML's
#: native ANE precision, so narrowing generation to avoid it would have thrown away the most
#: representative dtype on this target. iOS17 rather than the bare iOS16 minimum because it
#: also widens the supported op set, and nothing here needs to run on an older OS.
MINIMUM_DEPLOYMENT_TARGET = "iOS17"


def _compile_specs():
    """Compile specs for the partitioner: the deployment target above, otherwise defaults."""
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend as _CoreMLBackend

    return _CoreMLBackend.generate_compile_specs(
        minimum_deployment_target=getattr(ct.target, MINIMUM_DEPLOYMENT_TARGET),
    )


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

        config = LinearQuantizerConfig.from_dict({
            "global_config": {
                "quantization_scheme": QuantizationScheme.symmetric,
                "milestones": [0, 0, 10, 10],
                # coremltools validates this: activation_dtype must be quint8 or float32,
                # and rejects qint8 outright ("'activation_dtype' must be in
                # [torch.quint8, torch.float32]"). It raised while STRUCTURING the config,
                # so every quantized job died before lowering even started. Weights are
                # signed int8, activations unsigned — the asymmetry is CoreML's, not a typo.
                "activation_dtype": "quint8",
                "weight_dtype": "qint8",
            }
        })
        return CoreMLQuantizer(config)

    def _lower(self, ep, example_inputs):
        from executorch.backends.apple.coreml.partition.coreml_partitioner import (
            CoreMLPartitioner,
        )
        return lower_with_partitioner(ep, CoreMLPartitioner(compile_specs=_compile_specs()))
