"""qualcomm — QNN / Hexagon NPU (HTP).

Owns everything QNN-specific so it stays out of the shared path: the target SoC (an HTP
context binary is compiled for one Snapdragon arch), the fp16 HTP compiler spec, the
safe-op-support patch below, and QnnQuantizer for the int8 path.
"""

from __future__ import annotations

import os

from mobile.generator.lower.backends.base import EDGE_CONFIG, Backend, QuantMode

#: The HTP context binary is compiled for a specific Snapdragon SoC / HTP arch and must match
#: the phone the corpus runs on. Override with $QNN_SOC (SM8450 = 8 Gen 1, SM8475 = 8+ Gen 1,
#: both HTP v69).
QNN_SOC = os.environ.get("QNN_SOC", "SM8750")

#: Float fuzzing targets fp16 HTP; the quantized path uses QnnQuantizer instead.
USE_FP16 = True

_safe_op_support_installed = False


def _install_safe_op_support() -> None:
    """Make a QNN builder that THROWS during partition report "unsupported" instead.

    The partitioner probes support by actually calling each op's `define_node` builder. On
    fuzzer graphs those builders raise on shapes QNN never sees in real models — layer_norm
    with a None weight, sum over None dims, 0-d mean, complex64, an empty index list — and the
    raw exception propagates out of `to_backend`, losing the ENTIRE .pte over one un-buildable
    node. A static supported-op list cannot prevent this: the op IS supported in general, only
    this particular shape is not. Converting the raise into an ordinary "unsupported" verdict
    leaves that node on a portable CPU kernel and turns a whole-graph loss into ordinary
    partial delegation — measured to roughly double the lowered-graph yield, 27% to 49% on
    8-node graphs.

    This also subsumes ops QNN has no builder for at all: that lookup raises KeyError
    (node_visitors is a plain dict), which the same except clause turns into "unsupported".
    Idempotent.
    """
    global _safe_op_support_installed
    if _safe_op_support_installed:
        return
    from executorch.backends.qualcomm.partition import qnn_partitioner

    original = qnn_partitioner.QnnOperatorSupport.is_node_supported

    def safe_is_node_supported(self, submodules, node):
        try:
            return original(self, submodules, node)
        except Exception:
            return False

    qnn_partitioner.QnnOperatorSupport.is_node_supported = safe_is_node_supported
    _safe_op_support_installed = True


class QualcommBackend(Backend):
    name = "qualcomm"
    runs_on_host = False
    quant = QuantMode.OPTIONAL        # fp16 HTP, or int8 with --quantize

    def is_available(self) -> bool:
        try:
            import executorch.backends.qualcomm.partition.qnn_partitioner  # noqa: F401
            return True
        except Exception:
            return False

    def _compiler_specs(self):
        from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
        from executorch.backends.qualcomm.utils.utils import (
            generate_htp_compiler_spec,
            generate_qnn_executorch_compiler_spec,
        )

        return generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset[QNN_SOC],
            backend_options=generate_htp_compiler_spec(use_fp16=USE_FP16),
        )

    def quantizer(self):
        from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer

        return QnnQuantizer()

    def _lower(self, ep, example_inputs):
        # Generic to_edge + to_backend, NOT to_edge_transform_and_lower_to_qnn. Measured
        # head to head, the generic path yields MORE lowered graphs (53% vs 42%): the
        # dedicated path's QNN transform passes (DecomposeRoll, LayoutTransform, ...) crash on
        # more fuzzer graphs than the partition-boundary NoneType meta they avoid.
        from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
        from executorch.exir import to_edge

        _install_safe_op_support()
        edge = to_edge(ep, compile_config=EDGE_CONFIG)
        return edge.to_backend(QnnPartitioner(self._compiler_specs())).to_executorch()
