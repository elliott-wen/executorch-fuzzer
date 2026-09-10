"""ethos-u (Arm) — TOSA(INT) → Vela → Ethos-U command stream.

Ethos-U is an integer-only NPU: a network MUST be PT2E-quantized (TOSA INT profile) before
Vela can compile it into a command stream. So this backend ALWAYS quantizes — `lower` ignores
the --quantize flag, because an unquantized graph simply cannot target Ethos-U, and the fp32
eager oracle is never the right reference (see base.quantized_reference).

Distinct from a generic TOSA lowering: this pins a concrete Ethos-U target through
EthosUCompileSpec and runs the Vela compiler, producing a device-loadable .pte.

Lowers on x86; the .pte runs on Ethos-U hardware or the Corstone FVP simulator.
"""

from __future__ import annotations

from mobile.generator.lower.backends.base import Backend, QuantMode, lower_with_partitioner

#: Accelerator configuration. "ethos-u55-128" is an Ethos-U55 at 128 MACs (Corstone-300);
#: "ethos-u85-256" and friends retarget. system_config / memory_mode default from the target.
ETHOSU_TARGET = "ethos-u55-128"


def _compile_spec():
    from executorch.backends.arm.ethosu import EthosUCompileSpec

    return EthosUCompileSpec(target=ETHOSU_TARGET)


class EthosUBackend(Backend):
    name = "ethos-u"
    runs_on_host = False
    quant = QuantMode.ALWAYS          # integer-only NPU: every job is quantized

    def is_available(self) -> bool:
        try:
            import ethosu.vela  # noqa: F401 — Vela must import for lowering to work
            from executorch.backends.arm.ethosu import (  # noqa: F401
                EthosUCompileSpec,
                EthosUPartitioner,
            )
            return True
        except Exception:
            return False

    def quantizer(self):
        from executorch.backends.arm.quantizer import (
            EthosUQuantizer,
            get_symmetric_quantization_config,
        )
        quantizer = EthosUQuantizer(_compile_spec())
        quantizer.set_global(get_symmetric_quantization_config())
        return quantizer

    def _lower(self, ep, example_inputs):
        from executorch.backends.arm.ethosu import EthosUPartitioner

        return lower_with_partitioner(ep, EthosUPartitioner(_compile_spec()))
