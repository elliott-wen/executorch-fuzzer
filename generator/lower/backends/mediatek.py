"""mediatek — NeuroPilot APU (Dimensity NPU) delegate.

Ops the APU supports are delegated, the rest stay portable. There is no static supported-ops
list to maintain: NeuropilotOperatorsSupport asks mtk_converter's importer at partition time.
The one backend-specific knob is the target SoC, passed as the "platform-config" CompileSpec.

Supports PT2E quantization (int8). NeuropilotQuantizer is a torchao Quantizer subclass, so it
drops straight into the shared flow; its default precision is A8W8.

Lowers only on a host with the NeuroPilot Express SDK (mtk_converter + mtk_neuron wheels),
which ships as a cp310 wheel — so this lane runs in .venv-mtk, not the main venv. Point the
fleet at it with --python. The .pte executes on Dimensity 9300/9400 hardware, never on x86.
"""

from __future__ import annotations

from mobile.generator.lower.backends.base import Backend, QuantMode, lower_with_partitioner

#: Platform the APU graph compiles for. mt6989 = Dimensity 9300, mt6991 = Dimensity 9400.
#: Must match the device the corpus runs on.
PLATFORM_CONFIG = "mt6989"

_safe_op_support_installed = False


def _install_safe_op_support() -> None:
    """Make a converter probe that THROWS report "unsupported" instead.

    The partitioner asks mtk_converter whether each node is APU-supported. On fuzzer graphs
    that probe raises on shapes real models never have (KeyError: torch.float16, ValueError:
    Unsupported alpha input value, ...) and the exception propagates out of
    to_edge_transform_and_lower, losing the ENTIRE .pte over one un-probeable node. Catching
    it turns that whole-graph loss into ordinary partial delegation — the same fix as
    qualcomm._install_safe_op_support.

    Does NOT catch mtk_neuron compile errors raised later, in preprocess on the already-chosen
    partition; those still fail the graph, like QNN's context-binary errors. Idempotent.
    """
    global _safe_op_support_installed
    if _safe_op_support_installed:
        return
    from executorch.backends.mediatek import partitioner

    original = partitioner.NeuropilotOperatorsSupport.is_node_supported

    def safe_is_node_supported(self, submodules, node):
        try:
            return original(self, submodules, node)
        except Exception:
            return False

    partitioner.NeuropilotOperatorsSupport.is_node_supported = safe_is_node_supported
    _safe_op_support_installed = True


class MediatekBackend(Backend):
    name = "mediatek"
    runs_on_host = False
    quant = QuantMode.OPTIONAL        # float APU, or int8 with --quantize

    def is_available(self) -> bool:
        try:
            import mtk_converter  # noqa: F401 — the NeuroPilot converter must import
            from executorch.backends.mediatek import NeuropilotPartitioner  # noqa: F401
            return True
        except Exception:
            return False

    def quantizer(self):
        from executorch.backends.mediatek import NeuropilotQuantizer

        return NeuropilotQuantizer()

    def _lower(self, ep, example_inputs):
        from executorch.backends.mediatek import NeuropilotPartitioner
        from executorch.exir.backend.backend_details import CompileSpec

        _install_safe_op_support()
        return lower_with_partitioner(
            ep,
            NeuropilotPartitioner([CompileSpec("platform-config", PLATFORM_CONFIG.encode())]),
        )
