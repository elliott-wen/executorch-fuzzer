"""nxp — eIQ Neutron NPU: int8 graphs → Neutron converter → Neutron-C command stream.

The eIQ Neutron N3 (on the i.MX RT700 MCU) is integer-only, so a network MUST be
PT2E-quantized before the converter can compile it. Like ethos-u and cortex-m this backend
ALWAYS quantizes, and the fp32 eager oracle is never the right reference.

Lowering needs three things beyond the generic partitioner path, mirroring executorch's
examples/nxp/aot_neutron_compile.py: the Neutron edge passes as transform_passes, a
_core_aten_ops_exception_list on the EdgeCompileConfig (Neutron keeps aten.prelu), and the
POST-quantization state_dict handed to the partitioner so it can fold qparams.

Lowers on x86; the .pte runs on the i.MX RT700. Available only where the Neutron
partitioner/quantizer and the eIQ Neutron SDK import — the SDK is a private PyPI package
(`pip install --index-url https://eiq.nxp.com/repository eiq-neutron-sdk`, or executorch's
examples/nxp/setup.sh). Without it every import raises and the backend is simply unavailable.
"""

from __future__ import annotations

from mobile.generator.lower.backends.base import EDGE_CONFIG, Backend, QuantMode

#: Neutron target platform. imxrt700 (eIQ Neutron N3-64) is the only NPU the converter
#: currently supports.
NXP_TARGET = "imxrt700"


class NxpBackend(Backend):
    name = "nxp"
    runs_on_host = False
    quant = QuantMode.ALWAYS          # integer-only NPU: every job is quantized

    #: NeutronTargetSpec construction verifies the target against the SDK and is reused by
    #: both the quantizer and the partitioner, so it is built once per instance.
    _target_spec = None

    def _neutron_target_spec(self):
        if self._target_spec is None:
            from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec

            self._target_spec = NeutronTargetSpec(target=NXP_TARGET)
        return self._target_spec

    def is_available(self) -> bool:
        try:
            # Either import pulls in neutron_converter_manager, which raises unless the eIQ
            # Neutron SDK is installed — so this doubles as the SDK-presence check.
            from executorch.backends.nxp.neutron_partitioner import (  # noqa: F401
                NeutronPartitioner,
            )
            from executorch.backends.nxp.quantizer.neutron_quantizer import (  # noqa: F401
                NeutronQuantizer,
            )
            return True
        except Exception:
            return False

    def quantizer(self):
        from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer

        # NeutronQuantizer self-configures its per-op qconfigs from the target spec, so no
        # set_global is needed (unlike the Arm quantizers). is_qat=False selects PTQ.
        return NeutronQuantizer(self._neutron_target_spec(), is_qat=False)

    @staticmethod
    def _strict(ep, example_inputs):
        """Re-export under STRICT tracing.

        The NeutronQuantizer's conv/linear patterns only match the strict-export graph shape.
        Under a non-strict ep — which the shared pipeline produces — prepare_pt2e never
        annotates the weights, so the QDQ cluster never forms and NOTHING delegates: every
        graph silently falls back to portable. Strict re-export restores per-channel weight
        quantization and real Neutron delegation. A graph that exported non-strict but cannot
        export strict raises here and the job skips, which is correct — it simply is not
        tested under NXP.
        """
        from torch.export import export

        return export(ep.module(check_guards=False), example_inputs, strict=True)

    def lower(self, ep, example_inputs, quantize: bool = False):
        # Normalize to a strict ep before the shared PT2E flow. ALWAYS-quant, so base.lower
        # quantizes regardless of the flag.
        return super().lower(self._strict(ep, example_inputs), example_inputs, quantize)

    def quantized_reference(self, ep, example_inputs) -> list:
        # The reference must be quantized the SAME way as the delegated .pte, so it needs the
        # identical strict-export front end — otherwise it would carry the weight-unquantized
        # non-strict numerics and diverge from the device by construction.
        return super().quantized_reference(self._strict(ep, example_inputs), example_inputs)

    def _lower(self, ep, example_inputs):
        from executorch.backends.nxp.edge_passes.neutron_edge_pass_manager import (
            NeutronEdgePassManager,
        )
        from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
        from executorch.backends.nxp.nxp_backend import (
            core_aten_ops_exception_list,
            generate_neutron_compile_spec,
        )
        from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

        # `ep` is already PT2E-converted and re-exported (quant is ALWAYS), so ep.state_dict is
        # the post-quantization state dict the partitioner needs to fold quant params onto
        # delegated weights. On an ExportedProgram `state_dict` is a property holding a plain
        # dict, NOT a callable like nn.Module.state_dict() — the aot example calls it on a
        # module, we have an ep.
        partitioner = NeutronPartitioner(
            generate_neutron_compile_spec(NXP_TARGET),
            self._neutron_target_spec(),
            post_quantization_state_dict=ep.state_dict,
        )
        return to_edge_transform_and_lower(
            ep,
            transform_passes=NeutronEdgePassManager(),
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=EDGE_CONFIG._check_ir_validity,
                _core_aten_ops_exception_list=core_aten_ops_exception_list,
            ),
        ).to_executorch()
