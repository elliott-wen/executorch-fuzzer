"""openvino — Intel CPU/GPU/NPU delegate via the OpenVINO runtime.

Lowers on x86 and, for the CPU device, also RUNS there: the .pte's OpenVINO delegate blobs
execute in-process once libopenvino_backend is registered in the ExecuTorch runtime, exactly
like XNNPACK. Partial-delegates; there is no static supported-ops list to maintain, because
OpenvinoOperatorsSupport queries OpenVINO's own op-support table at partition time. The one
backend-specific knob is the target DEVICE, passed as a single CompileSpec blob.

Supports PT2E quantization (int8). OpenVINOQuantizer is a torchao Quantizer subclass, so it
drops into the shared flow — but unlike XNNPACK and Ethos-U it is configured by a `mode=`
constructor argument rather than set_global(), so quantizer() just constructs it (default
mode INT8_SYM).

Install with `pip install executorch[openvino]`, which also pulls nncf for the quantizer.

Note for anyone running a full sweep: co-loading this with QNN in one process corrupts the
heap during OpenVINO lowering. The registry's lazy probing keeps that from happening by
accident; MOBILE_BACKENDS=openvino is the belt-and-braces guard.
"""

from __future__ import annotations

import os

from mobile.generator.lower.backends.base import Backend, QuantMode, lower_with_partitioner

#: Device the OpenVINO graph compiles for: CPU | GPU | NPU. CPU is the only one that runs on a
#: generic x86 host; GPU/NPU need Intel hardware and drivers. The choice is baked into the
#: .pte, so corpora are device-specific.
OPENVINO_DEVICE = os.environ.get("OPENVINO_DEVICE", "CPU").upper()


class OpenVINOBackend(Backend):
    name = "openvino"
    runs_on_host = True
    # Declared as a QuantMode, never as a separate boolean. This class once carried a bare
    # `supports_quantization = True` alongside `quant = NEVER`: the pipeline read the boolean
    # and believed the backend could quantize, then base.lower refused every job outright.
    # QuantMode is the single source of truth, and base.quantizes() is how to ask.
    quant = QuantMode.OPTIONAL

    def is_available(self) -> bool:
        try:
            import openvino  # noqa: F401 — the runtime must import for lowering
            from executorch.backends.openvino.partitioner import (  # noqa: F401
                OpenvinoPartitioner,
            )
            return True
        except Exception:
            return False

    def quantizer(self):
        from executorch.backends.openvino.quantizer import OpenVINOQuantizer

        return OpenVINOQuantizer()

    def _lower(self, ep, example_inputs):
        from executorch.backends.openvino.partitioner import OpenvinoPartitioner
        from executorch.exir.backend.backend_details import CompileSpec

        return lower_with_partitioner(
            ep, OpenvinoPartitioner([CompileSpec("device", OPENVINO_DEVICE.encode())])
        )
