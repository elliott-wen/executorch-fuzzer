"""NXP eIQ Neutron backend — int8 graphs → Neutron converter → Neutron-C command stream.

The Neutron NPU (eIQ Neutron N3, on the i.MX RT700 MCU) is an integer-only accelerator:
a network MUST be PT2E-quantized before the Neutron converter can compile it. So like
ethos-u/cortex-m this backend ALWAYS quantizes — `base.lower()` runs the PT2E flow
unconditionally and the fp32 eager oracle is never the right reference (pregen stores a
quantized reference instead; see backends/base.quantized_reference).

Unlike the generic partitioner backends, NXP lowering needs three extra things (mirroring
executorch's examples/nxp/aot_neutron_compile.py), so `_lower` is overridden rather than
using the shared `lower_with_partitioner`:
  - the Neutron edge passes (NeutronEdgePassManager) run as `transform_passes`,
  - `_core_aten_ops_exception_list` on the EdgeCompileConfig (Neutron keeps aten.prelu),
  - the partitioner is handed the POST-quantization state_dict so it can fold qparams.

Lowers on the host (x86); the `.pte` runs on the i.MX RT700 device — `runs_on_host = False`.
Available only where the Neutron partitioner/quantizer AND the eIQ Neutron SDK import; the
SDK is a private PyPI package (`pip install --index-url https://eiq.nxp.com/repository
eiq-neutron-sdk`, or run executorch's examples/nxp/setup.sh). Without it every import here
raises "eIQ Neutron SDK not found" and the backend simply isn't registered.
"""

from __future__ import annotations

from .base import Backend, QuantMode, EDGE_CFG

# Neutron target platform. "imxrt700" (eIQ Neutron N3-64 on the i.MX RT700) is the only
# NPU currently supported by the Neutron converter.
NXP_TARGET = "imxrt700"


class NxpBackend(Backend):
    name = "nxp"
    runs_on_host = False
    quant = QuantMode.ALWAYS          # integer-only NPU: every job is quantized

    # NeutronTargetSpec construction verifies the target against the SDK (and is reused by
    # both the quantizer and the partitioner), so build it once and cache it per-instance.
    _target_spec = None

    def _neutron_target_spec(self):
        if self._target_spec is None:
            from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
            self._target_spec = NeutronTargetSpec(target=NXP_TARGET)
        return self._target_spec

    def is_available(self) -> bool:
        try:
            # Importing either of these pulls in neutron_converter_manager, which raises
            # unless the eIQ Neutron SDK (eiq_neutron_sdk) is installed — so this is also
            # the SDK-presence check.
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
        # NeutronQuantizer self-configures its per-op qconfigs from the target spec; no
        # set_global is needed (unlike the Arm quantizers). is_qat=False → PTQ.
        return NeutronQuantizer(self._neutron_target_spec(), is_qat=False)

    @staticmethod
    def _strict(ep, example_inputs):
        """Re-export `ep` under STRICT tracing. The NeutronQuantizer's conv/linear
        patterns only match the strict-export graph shape; under a non-strict ep (which
        the shared pregen pipeline produces) prepare_pt2e never annotates the weights, so
        the QDQ cluster never forms and NOTHING delegates — every graph silently falls
        back to portable (ops=0). Strict re-export restores per-channel weight quant and
        real Neutron delegation. A graph that non-strict-exported but can't strict-export
        simply raises here → the job SKIPs (correct: we just don't test it under NXP)."""
        from torch.export import export
        return export(ep.module(check_guards=False), example_inputs, strict=True)

    def lower(self, ep, example_inputs, quantize: bool = False):
        # Normalize to a strict ep before the shared PT2E flow (see _strict). ALWAYS-quant,
        # so base.lower quantizes regardless of the flag.
        return super().lower(self._strict(ep, example_inputs), example_inputs, quantize)

    def quantized_reference(self, ep, example_inputs) -> list:
        # The oracle must be quantized the SAME way as the delegated .pte, so it needs the
        # identical strict-export front end — otherwise the reference would be the (weight-
        # unquantized) non-strict numerics and diverge from the device by construction.
        return super().quantized_reference(self._strict(ep, example_inputs), example_inputs)

    def _lower(self, ep, example_inputs):
        from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
        from executorch.backends.nxp.neutron_partitioner import NeutronPartitioner
        from executorch.backends.nxp.nxp_backend import (
            core_aten_ops_exception_list,
            generate_neutron_compile_spec,
        )
        from executorch.backends.nxp.edge_passes.neutron_edge_pass_manager import (
            NeutronEdgePassManager,
        )

        target_spec = self._neutron_target_spec()
        compile_spec = generate_neutron_compile_spec(NXP_TARGET)
        # `ep` is already PT2E-converted + re-exported (base.lower quantized it because
        # quant is ALWAYS), so ep.state_dict is the post-quantization state dict the
        # partitioner needs to fold quant params onto delegated weights. Note: on an
        # ExportedProgram `state_dict` is a property (a plain dict), NOT a callable method
        # like nn.Module.state_dict() — the aot example calls it on a module, we have an ep.
        partitioner = NeutronPartitioner(
            compile_spec,
            target_spec,
            post_quantization_state_dict=ep.state_dict,
        )
        return to_edge_transform_and_lower(
            ep,
            transform_passes=NeutronEdgePassManager(),
            partitioner=[partitioner],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=EDGE_CFG._check_ir_validity,
                _core_aten_ops_exception_list=core_aten_ops_exception_list,
            ),
        ).to_executorch()
