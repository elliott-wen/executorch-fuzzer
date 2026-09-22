"""samsung — Exynos ENN (NPU) delegate.

Ops the Exynos NPU has an ENN builder for, or that the ENN compiler reports as supported, are
delegated; the rest stay portable. No static supported-ops list: EnnOperatorSupport asks the
ENN wrapper and the per-op node_visitor table at partition time. The backend-specific knob is
the target CHIPSET, passed as the ENN compile-options CompileSpec.

Unlike the generic partitioner backends ENN needs its OWN edge config — `_skip_dim_order=True`
(the builders consume the legacy, non-dim-order ops) plus a core-aten exception list keeping
ops like linear/max_pool2d/layer_norm UN-decomposed so the builders can match them — and a
small transform pass list. So `_lower` replicates to_edge_transform_and_lower_to_enn rather
than using the shared helper: the passes and the partitioner must run inside a SINGLE
to_edge_transform_and_lower, or a to_edge/to_backend split would decompose those ops before
they reach the partition boundary.

Supports PT2E quantization (int8 A8W8); EnnQuantizer is a torchao Quantizer subclass with
per-channel weights by default.

Lowers only where the ENN SDK is installed — the compiled PyEnnWrapperAdaptor extension must
import. The SDK ships as a manual tarball with no pip package; setup_samsung_sdk.sh extracts
it, builds the extension into the executorch package, and patchelfs the SDK libs' RUNPATH to
$ORIGIN so they find their siblings at load time. With that done the extension imports lazily
with no LD_LIBRARY_PATH and no preloading — force-preloading every SDK lib pulls in ones that
interpose on torch's C++ runtime and abort with std::bad_cast.
"""

from __future__ import annotations

from mobile.generator.lower.backends.base import Backend, QuantMode

#: Exynos chipset the ENN graph compiles for; E9955 = Exynos 2500, E9965 = Exynos 2600.
#: Must match the device the corpus runs on. Case-insensitive.
CHIPSET = "E9965"


class SamsungBackend(Backend):
    name = "samsung"
    runs_on_host = False
    quant = QuantMode.OPTIONAL        # float NPU, or int8 A8W8 with --quantize

    def is_available(self) -> bool:
        try:
            # The partitioner imports the compiled PyEnnWrapperAdaptor extension; it must
            # load for lowering to work, so probe it explicitly too.
            import executorch.backends.samsung.python.PyEnnWrapperAdaptor  # noqa: F401
            from executorch.backends.samsung.partition.enn_partitioner import (  # noqa: F401
                EnnPartitioner,
            )
            return True
        except Exception:
            return False

    def quantizer(self):
        from executorch.backends.samsung.quantizer.quantizer import EnnQuantizer

        return EnnQuantizer()

    def _edge_config(self):
        """Keep the legacy (non-dim-order) ops the ENN builders match, and leave a handful of
        high-level ops un-decomposed so they reach a builder. Verifier disabled as elsewhere,
        to recover graphs the Edge spec would reject but the runtime would run."""
        from executorch.exir import EdgeCompileConfig
        from executorch.exir.dialects._ops import ops as exir_ops

        return EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
            _core_aten_ops_exception_list=[
                exir_ops.edge.aten.max_pool2d.default,
                exir_ops.edge.aten.linear.default,
                exir_ops.edge.aten.hardswish.default,
                exir_ops.edge.aten.prelu.default,
                exir_ops.edge.aten.pixel_shuffle.default,
                exir_ops.edge.aten._safe_softmax.default,
                exir_ops.edge.aten.layer_norm.default,
                exir_ops.edge.aten.matmul.default,
                exir_ops.edge.aten.hardsigmoid.default,
            ],
        )

    def _lower(self, ep, example_inputs):
        # executorch 1.5.0 renamed this pass: _passes/fuse_conv_act.FuseConvActPass became
        # _passes/fuse_activation.FuseActivationPass. Same pass, wider remit (it fuses
        # activations beyond conv). Try the new name first and fall back, so this adapter
        # works on either wheel — on 1.5.1 the old import raised ModuleNotFoundError inside
        # _lower and took EVERY samsung graph with it (0/300 lowered, no other symptom).
        try:
            from executorch.backends.samsung._passes.fuse_activation import (
                FuseActivationPass as FuseConvActPass,
            )
        except ModuleNotFoundError:
            from executorch.backends.samsung._passes.fuse_conv_act import FuseConvActPass
        from executorch.backends.samsung._passes.remove_useless_ops import RemoveUselessOpPass
        from executorch.backends.samsung.partition.enn_partitioner import EnnPartitioner
        from executorch.backends.samsung.serialization.compile_options import (
            gen_samsung_backend_compile_spec,
        )
        from executorch.backends.transforms.remove_clone_ops import RemoveCloneOpsTransform
        from executorch.exir.program._program import to_edge_transform_and_lower

        compile_specs = [gen_samsung_backend_compile_spec(CHIPSET)]
        passes = [RemoveUselessOpPass(), RemoveCloneOpsTransform(), FuseConvActPass()]
        return to_edge_transform_and_lower(
            ep,
            passes,
            {"forward": [EnnPartitioner(compile_specs)]},
            compile_config=self._edge_config(),
        ).to_executorch()
