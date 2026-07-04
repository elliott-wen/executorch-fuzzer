"""Samsung Exynos / ENN (NPU) backend — Exynos AI accelerator delegate.

Lowers via the EnnPartitioner: ops the Exynos NPU has an ENN builder for (or that the ENN
compiler reports as supported) are delegated, the rest stay portable. There's no static
supported-ops list to maintain here — EnnOperatorSupport asks the ENN wrapper / the per-op
node_visitor table at partition time. The one backend-specific knob is the target CHIPSET,
passed as the ENN compile-options CompileSpec via gen_samsung_backend_compile_spec.

Unlike the generic partitioner backends, ENN needs its OWN edge config: `_skip_dim_order=True`
(the ENN builders consume the legacy, non-dim-order ops) plus a core-aten exception list that
keeps ops like linear/max_pool2d/layer_norm UN-decomposed so the builders can match them — and
a small transform pass list (remove-useless / remove-clone / fuse-conv-act). So `_lower`
replicates the canonical to_edge_transform_and_lower_to_enn flow rather than the shared helper.

Supports PT2E quantization (int8 A8W8): EnnQuantizer is a torchao Quantizer subclass
(implements annotate), so it drops straight into the shared _quantize_pt2e flow. Default
precision is A8W8, per-channel weights — quantizer() just returns EnnQuantizer().

Lowers only where the ENN SDK is installed (the compiled PyEnnWrapperAdaptor extension must
import); the .pte executes on Exynos NPU hardware, never on the x86 host — so
runs_on_host = False.
"""

from __future__ import annotations

from .base import Backend, QuantMode

# Targeted Exynos chipset the ENN graph compiles for; one of SamsungChipset (E9955 = Exynos
# 2500, E9965 = Exynos 2600). Must match the device the corpus runs on. Case-insensitive.
#
# SDK note: the Exynos AI LiteCore SDK ships as a manual tarball (no pip package);
# setup_samsung_sdk.sh extracts it, builds the PyEnnWrapperAdaptor extension into the
# executorch package, and patchelfs the SDK libs' RUNPATH to $ORIGIN so they find their
# siblings at load time. With that in place the extension imports lazily with no
# LD_LIBRARY_PATH and no preloading (force-preloading every SDK lib pulls in ones that
# interpose on torch's C++ runtime and abort with std::bad_cast).
CHIPSET = "E9965"


class SamsungBackend(Backend):
    name = "samsung"
    runs_on_host = False
    quant = QuantMode.OPTIONAL        # float NPU, or int8 A8W8 with --quantize (EnnQuantizer)

    def is_available(self) -> bool:
        try:
            from executorch.backends.samsung.partition.enn_partitioner import (  # noqa: F401
                EnnPartitioner,
            )
            # The partitioner imports the compiled PyEnnWrapperAdaptor extension; it must load
            # for lowering to work, so probe it here too.
            import executorch.backends.samsung.python.PyEnnWrapperAdaptor  # noqa: F401
            return True
        except Exception:
            return False

    def quantizer(self):
        from executorch.backends.samsung.quantizer.quantizer import EnnQuantizer
        return EnnQuantizer()

    def _edge_config(self):
        """ENN's edge config: keep the legacy (non-dim-order) ops the builders match, and
        leave a handful of high-level ops UN-decomposed so they reach an ENN builder. Verifier
        disabled like the rest of the framework to recover graphs the Edge spec would reject."""
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
        # Mirror to_edge_transform_and_lower_to_enn: the ENN transform passes + EnnPartitioner
        # run inside a single to_edge_transform_and_lower so the non-decomposed ops survive to
        # the partition boundary (a to_edge/to_backend split would decompose them first).
        from executorch.exir.program._program import to_edge_transform_and_lower
        from executorch.backends.samsung.partition.enn_partitioner import EnnPartitioner
        from executorch.backends.samsung.serialization.compile_options import (
            gen_samsung_backend_compile_spec,
        )
        from executorch.backends.samsung._passes.fuse_conv_act import FuseConvActPass
        from executorch.backends.samsung._passes.remove_useless_ops import (
            RemoveUselessOpPass,
        )
        from executorch.backends.transforms.remove_clone_ops import (
            RemoveCloneOpsTransform,
        )

        compile_specs = [gen_samsung_backend_compile_spec(CHIPSET)]
        passes = [RemoveUselessOpPass(), RemoveCloneOpsTransform(), FuseConvActPass()]
        return to_edge_transform_and_lower(
            ep,
            passes,
            {"forward": [EnnPartitioner(compile_specs)]},
            compile_config=self._edge_config(),
        ).to_executorch()
