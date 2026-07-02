"""MediaTek / NeuroPilot (APU) backend — Dimensity NPU delegate.

Lowers via the NeuropilotPartitioner: ops the APU supports are delegated, the rest stay
portable. There's no static supported-ops list to maintain — NeuropilotOperatorsSupport asks
mtk_converter's importer (importer_v2.is_fx_node_supported) at partition time. The one
backend-specific knob is the target SoC, passed as the "platform-config" CompileSpec.

Supports PT2E quantization (int8): NeuropilotQuantizer is a torchao Quantizer subclass
(implements annotate), so it drops straight into the shared _quantize_pt2e flow. Its default
precision is A8W8; setup_precision(Precision.X) selects a different one (A16W16 / A16W8 / ...).

Lowers and runs only on a MediaTek host with the NeuroPilot Express SDK installed
(mtk_converter + mtk_neuron wheels); the .pte executes on Dimensity 9300/9400 hardware, never
on the x86 host — so runs_on_host = False. Available only where the partitioner AND
mtk_converter import.
"""

from __future__ import annotations

from .base import Backend, QuantMode, lower_with_partitioner

# Targeted MediaTek platform the APU graph compiles for. mt6989 = Dimensity 9300; mt6991 =
# Dimensity 9400. Must match the device the corpus runs on.
PLATFORM_CONFIG = "mt6989"


_SAFE_OP_SUPPORT_INSTALLED = False


def _install_safe_op_support() -> None:
    """Make NeuropilotOperatorsSupport.is_node_supported report 'unsupported' (→ node stays on
    a portable CPU kernel) when the converter's probe THROWS, instead of letting the exception
    abort the whole graph's lowering.

    The partitioner asks mtk_converter (importer_v2.is_fx_node_supported) whether each node is
    APU-supported. On fuzzer graphs that probe raises on shapes real models never have —
    KeyError: torch.float16, ValueError: Unsupported alpha input value, etc. — and the raw
    exception propagates out of to_edge_transform_and_lower, losing the ENTIRE .pte over one
    un-probeable node. Wrapping it in try/except converts that whole-graph loss into ordinary
    partial delegation. Same fix as the QNN backend (see qualcomm._install_safe_op_support).
    Idempotent. (Does NOT catch mtk_neuron compile errors raised later in preprocess on the
    already-chosen partition — those still fail the graph, like QNN's context-binary errors.)"""
    global _SAFE_OP_SUPPORT_INSTALLED
    if _SAFE_OP_SUPPORT_INSTALLED:
        return
    from executorch.backends.mediatek import partitioner as mp

    orig = mp.NeuropilotOperatorsSupport.is_node_supported

    def safe_is_node_supported(self, submodules, node):
        try:
            return orig(self, submodules, node)
        except Exception:
            return False

    mp.NeuropilotOperatorsSupport.is_node_supported = safe_is_node_supported
    _SAFE_OP_SUPPORT_INSTALLED = True


class MediatekBackend(Backend):
    name = "mediatek"
    runs_on_host = False
    quant = QuantMode.OPTIONAL        # float APU, or int8 with --quantize (NeuropilotQuantizer)

    def is_available(self) -> bool:
        try:
            from executorch.backends.mediatek import (  # noqa: F401
                NeuropilotPartitioner,
            )
            import mtk_converter  # noqa: F401 — the NeuroPilot converter must import to lower
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
