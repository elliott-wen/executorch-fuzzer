"""QNN / Hexagon-NPU (HTP) backend.

Owns everything QNN-specific so it stays out of the shared path:
  - target SoC (the HTP context binary is compiled for a specific Snapdragon arch),
  - the fp16 HTP compiler spec,
  - the safe-op-support patch (see _install_safe_op_support) — converts a QNN builder that
    THROWS during partition into an ordinary "unsupported" verdict, so one un-buildable node
    stays a portable CPU kernel instead of sinking the whole graph,
  - QnnQuantizer for the quantized (int8) path.
"""

from __future__ import annotations

import os

from .base import Backend, QuantMode, EDGE_CFG

# HTP context binary is compiled for a specific Snapdragon SoC / HTP arch — must match the
# phone the corpus runs on. SM8450 = Snapdragon 8 Gen 1 (HTP v69). Override with $QNN_SOC
# (e.g. SM8475 = ROG Phone 6 / Snapdragon 8+ Gen 1, same v69 arch). Change default here to retarget.
QNN_SOC = os.environ.get("QNN_SOC", "SM8450")

# Float fuzzer → fp16 HTP. The quantized path (supports_quantization) uses QnnQuantizer instead.
USE_FP16 = True


_SAFE_OP_SUPPORT_INSTALLED = False


def _install_safe_op_support() -> None:
    """Make QnnOperatorSupport.is_node_supported report 'unsupported' (→ node stays on a
    portable CPU kernel) when a QNN builder THROWS, instead of letting the exception abort
    the whole graph's lowering.

    The partitioner probes support by actually calling each op's `define_node` builder
    (qnn_partitioner.is_node_supported). On fuzzer graphs those builders raise on shapes QNN
    never sees in real models — layer_norm with a None weight, sum over None dims, 0-d mean,
    complex64, an empty index list — and the raw exception propagates out of `to_backend`,
    losing the ENTIRE .pte over one un-buildable node. A static supported-op list can't
    prevent this: the op IS supported in general, only this particular shape isn't. Wrapping
    the probe in try/except converts that whole-graph loss into ordinary partial delegation
    and roughly doubles the lowered-graph yield (~27% → ~49% on 8-node graphs).

    This also subsumes skipping ops QNN has no builder for at all: that lookup raises
    KeyError (node_visitors is a plain dict), which the same except clause turns into
    'unsupported'. Idempotent."""
    global _SAFE_OP_SUPPORT_INSTALLED
    if _SAFE_OP_SUPPORT_INSTALLED:
        return
    from executorch.backends.qualcomm.partition import qnn_partitioner as qp

    orig = qp.QnnOperatorSupport.is_node_supported

    def safe_is_node_supported(self, submodules, node):
        try:
            return orig(self, submodules, node)
        except Exception:
            return False

    qp.QnnOperatorSupport.is_node_supported = safe_is_node_supported
    _SAFE_OP_SUPPORT_INSTALLED = True


class QualcommBackend(Backend):
    name = "qualcomm"
    runs_on_host = False
    quant = QuantMode.OPTIONAL        # fp16 HTP, or int8 with --quantize (QnnQuantizer)

    def is_available(self) -> bool:
        try:
            import executorch.backends.qualcomm.partition.qnn_partitioner  # noqa: F401
            return True
        except Exception:
            return False

    def _compiler_specs(self):
        from executorch.backends.qualcomm.utils.utils import (
            generate_qnn_executorch_compiler_spec,
            generate_htp_compiler_spec,
        )
        from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset

        return generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset[QNN_SOC],
            backend_options=generate_htp_compiler_spec(use_fp16=USE_FP16),
        )

    def quantizer(self):
        from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer
        return QnnQuantizer()

    def _lower(self, ep, example_inputs):
        # Generic to_edge + to_backend, NOT to_edge_transform_and_lower_to_qnn. Measured
        # head-to-head, the generic path yields MORE lowered graphs (~53% vs ~42%): the
        # dedicated path's QNN transform passes (DecomposeRoll/LayoutTransform/...) crash on
        # more fuzzer graphs than the partition-boundary NoneType meta they avoid.
        # _install_safe_op_support makes the partitioner keep any node QNN can't build (no
        # builder, or a builder that throws on this shape) on a portable CPU kernel.
        from executorch.exir import to_edge
        from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner

        _install_safe_op_support()
        edge = to_edge(ep, compile_config=EDGE_CFG)
        partitioner = QnnPartitioner(self._compiler_specs())
        return edge.to_backend(partitioner).to_executorch()
