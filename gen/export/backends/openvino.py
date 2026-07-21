"""OpenVINO backend — Intel CPU/GPU/NPU delegate via the OpenVINO runtime.

Lowers on the host (x86) and, for the CPU device, also RUNS on the host: the .pte's
OpenVINO delegate blobs are executed in-process by the ExecuTorch runtime once the OpenVINO
backend (libopenvino_backend) is registered in it — exactly like XNNPACK. So
`runs_on_host = True`; the openvino_client mirrors xnnpack_client (host runtime, no simulator).

Partial-delegates: ops OpenVINO doesn't support stay portable. Unlike qualcomm there's no
static supported-ops list to maintain — OpenvinoOperatorsSupport queries OpenVINO's own
op-support table at partition time. The only backend-specific knob is the target DEVICE,
passed as a single CompileSpec blob.

Supports PT2E quantization (int8): OpenVINOQuantizer is a torchao Quantizer subclass
(implements annotate), so it drops straight into the shared _quantize_pt2e flow. Unlike
XNNPACK/Ethos-U it's configured by the `mode=` ctor kwarg, not set_global() — so quantizer()
just returns OpenVINOQuantizer() (default mode INT8_SYM).

Available only where the OpenVINO partitioner AND the openvino runtime import
(`pip install executorch[openvino]`, which also pulls nncf for the quantizer).
"""

from __future__ import annotations

import os

from .base import Backend, lower_with_partitioner

# Target device the OpenVINO graph compiles for: "CPU" | "GPU" | "NPU". CPU is the only one
# that runs on a generic x86 host; GPU/NPU require Intel hardware + drivers. Override at
# lower time with OPENVINO_DEVICE=GPU (baked into the .pte, so corpora are device-specific).
OPENVINO_DEVICE = os.environ.get("OPENVINO_DEVICE", "CPU").upper()


class OpenVINOBackend(Backend):
    name = "openvino"
    runs_on_host = True
    supports_quantization = True

    def is_available(self) -> bool:
        try:
            from executorch.backends.openvino.partitioner import (  # noqa: F401
                OpenvinoPartitioner,
            )
            import openvino  # noqa: F401 — the OpenVINO runtime must import to lower
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
