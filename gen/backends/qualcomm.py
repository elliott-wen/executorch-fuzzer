"""QNN / Hexagon-NPU (HTP) backend.

Owns everything QNN-specific so it stays out of the shared path:
  - target SoC (the HTP context binary is compiled for a specific Snapdragon arch),
  - the fp16 HTP compiler spec,
  - SUPPORTED_OPS — a STATIC build list (the aten targets QNN has a builder for). Any op in
    a graph NOT in this list is added to skip_node_op_set so it stays a portable CPU kernel;
    otherwise QNN lowering THROWS on the first unsupported op and the whole graph is lost.
  - QnnQuantizer for the quantized (int8) path.

NOTE on the skip set: the partitioner matches skip names against the EDGE node name
(node.target.__name__, e.g. "aten.gt.Scalar"). The pre-edge exported program carries .out /
overload-variant names ("aten.gt.Scalar_out") that DON'T match, so the skip set MUST be
computed AFTER to_edge — that's why _lower runs to_edge itself before partitioning.
"""

from __future__ import annotations

from .base import Backend, EDGE_CFG

# HTP context binary is compiled for a specific Snapdragon SoC / HTP arch — must match the
# phone the corpus runs on. SM8450 = Snapdragon 8 Gen 1 (HTP v69). Change here to retarget.
QNN_SOC = "SM8450"

# Float fuzzer → fp16 HTP. The quantized path (supports_quantization) uses QnnQuantizer instead.
USE_FP16 = True

# Static build list: the aten targets QNN has a NodeVisitor builder for, in edge-name form
# (matches node.target.__name__ after to_edge). Regenerate with:
#   python -c "import executorch.backends.qualcomm.builders as b; \
#     from executorch.backends.qualcomm.builders import node_visitor_manager as m; \
#     print(sorted(m._node_visitor_dict))"
SUPPORTED_OPS = frozenset({
    "aten._adaptive_avg_pool3d.default", "aten._log_softmax.default",
    "aten._native_batch_norm_legit.no_stats",
    "aten._native_batch_norm_legit_no_training.default", "aten._safe_softmax.default",
    "aten._softmax.default", "aten._to_copy.default", "aten.abs.default",
    "aten.adaptive_avg_pool2d.default", "aten.adaptive_max_pool2d.default", "aten.add.Tensor",
    "aten.amax.default", "aten.amin.default", "aten.arange.start_step", "aten.argmax.default",
    "aten.argmin.default", "aten.asin.default", "aten.atan.default", "aten.avg_pool2d.default",
    "aten.avg_pool3d.default", "aten.bitwise_and.Tensor", "aten.bitwise_or.Tensor",
    "aten.bitwise_xor.Tensor", "aten.bmm.default", "aten.cat.default", "aten.ceil.default",
    "aten.channel_shuffle.default", "aten.clamp.default", "aten.constant_pad_nd.default",
    "aten.convolution.default", "aten.copy.default", "aten.cos.default", "aten.cumsum.default",
    "aten.div.Tensor", "aten.elu.default", "aten.embedding.default", "aten.eq.Tensor",
    "aten.exp.default", "aten.expand_copy.default", "aten.flip.default", "aten.floor.default",
    "aten.floor_divide.default", "aten.full.default", "aten.full_like.default",
    "aten.gather.default", "aten.ge.Tensor", "aten.gelu.default", "aten.grid_sampler_2d.default",
    "aten.grid_sampler_3d.default", "aten.gt.Tensor", "aten.hardsigmoid.default",
    "aten.hardswish.default", "aten.hardtanh.default", "aten.index.Tensor",
    "aten.index_put.default", "aten.index_select.default", "aten.instance_norm.default",
    "aten.isinf.default", "aten.isnan.default", "aten.le.Tensor", "aten.linear.default",
    "aten.log.default", "aten.logical_and.default", "aten.logical_not.default", "aten.lt.Tensor",
    "aten.matmul.default", "aten.max.dim", "aten.max_pool2d_with_indices.default",
    "aten.maximum.default", "aten.mean.dim", "aten.min.dim", "aten.minimum.default",
    "aten.mm.default", "aten.mul.Tensor", "aten.native_group_norm.default",
    "aten.native_layer_norm.default", "aten.ne.Tensor", "aten.neg.default",
    "aten.permute_copy.default", "aten.pixel_shuffle.default", "aten.pixel_unshuffle.default",
    "aten.pow.Tensor_Tensor", "aten.prelu.default", "aten.rand.default", "aten.rand_like.default",
    "aten.randn.default", "aten.randn_like.default", "aten.reciprocal.default",
    "aten.reflection_pad1d.default", "aten.reflection_pad2d.default", "aten.relu.default",
    "aten.repeat.default", "aten.rms_norm.default", "aten.round.default", "aten.rsqrt.default",
    "aten.scatter.src", "aten.select.int", "aten.select_copy.int", "aten.sigmoid.default",
    "aten.sign.default", "aten.sin.default", "aten.slice_copy.Tensor", "aten.slice_scatter.default",
    "aten.split_with_sizes.default", "aten.split_with_sizes_copy.default", "aten.sqrt.default",
    "aten.squeeze.dims", "aten.squeeze_copy.dims", "aten.stack.default", "aten.sub.Tensor",
    "aten.sum.dim_IntList", "aten.tanh.default", "aten.topk.default", "aten.unbind.int",
    "aten.unsqueeze_copy.default", "aten.upsample_bicubic2d.vec", "aten.upsample_bilinear2d.default",
    "aten.upsample_bilinear2d.vec", "aten.upsample_nearest2d.default", "aten.upsample_nearest2d.vec",
    "aten.view_copy.default", "aten.where.self", "dim_order_ops._to_dim_order_copy.default",
    "getitem", "quantized_decomposed.dequantize_per_channel.default",
    "quantized_decomposed.dequantize_per_channel.tensor",
    "quantized_decomposed.dequantize_per_tensor.default",
    "quantized_decomposed.dequantize_per_tensor.tensor",
    "quantized_decomposed.quantize_per_channel.default",
    "quantized_decomposed.quantize_per_tensor.default", "scalar_tensor.default",
})


class QualcommBackend(Backend):
    name = "qualcomm"
    runs_on_host = False
    supports_quantization = True

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

    def _skip_node_op_set(self, edge) -> set[str]:
        """Edge ops with no QNN builder → keep them OUT of the QNN partition (stay portable).
        Computed on the EDGE graph so names match node.target.__name__ at partition time (the
        pre-edge program carries .out-variant names that wouldn't match)."""
        nodes = edge.exported_program().graph_module.graph.nodes
        return {
            n.target.__name__
            for n in nodes
            if n.op == "call_function"
            and getattr(n.target, "__name__", None) not in SUPPORTED_OPS
        }

    def _lower(self, ep, example_inputs):
        # Generic to_edge + to_backend, NOT to_edge_transform_and_lower_to_qnn. Measured
        # head-to-head, the generic path yields MORE lowered graphs (~53% vs ~42%): the
        # dedicated path's QNN transform passes (DecomposeRoll/LayoutTransform/...) crash on
        # more fuzzer graphs than the partition-boundary NoneType meta they avoid.
        # skip_node_op_set keeps QNN-unsupported ops on portable CPU kernels.
        from executorch.exir import to_edge
        from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner

        edge = to_edge(ep, compile_config=EDGE_CFG)
        partitioner = QnnPartitioner(
            self._compiler_specs(),
            skip_node_op_set=self._skip_node_op_set(edge),
        )
        return edge.to_backend(partitioner).to_executorch()
