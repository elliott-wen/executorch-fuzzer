"""Arm Ethos-U backend — TOSA(INT) → Vela → Ethos-U command stream.

Ethos-U is an integer-only NPU: a network MUST be PT2E-quantized (TOSA INT profile)
before it can be compiled by Vela into an Ethos-U command stream. So unlike the float
backends, this one ALWAYS quantizes — `lower()` ignores the `quantize` flag and runs the
PT2E flow unconditionally (an unquantized graph simply cannot target Ethos-U).

Distinct from the `arm` backend: that one lowers to the generic TOSA dialect (no Vela,
no specific accelerator); this one pins a concrete Ethos-U target via EthosUCompileSpec
and runs the Vela compiler, producing a device-loadable `.pte`.

Lowers on the host (x86); the `.pte` runs on the Ethos-U device or the Corstone FVP
simulator — `runs_on_host = False`. Available only where the Arm Ethos-U partitioner AND
the Vela compiler import.
"""

from __future__ import annotations

from .base import Backend, lower_with_partitioner, _quantize_pt2e

# Ethos-U accelerator configuration. "ethos-u55-128" = Ethos-U55 @ 128 MACs (Corstone-300);
# "ethos-u85-256" etc. retargets. system_config / memory_mode default from the target.
ETHOSU_TARGET = "ethos-u55-128"


def _compile_spec():
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    return EthosUCompileSpec(target=ETHOSU_TARGET)


class EthosUBackend(Backend):
    name = "ethos-u"
    runs_on_host = False
    supports_quantization = True

    def is_available(self) -> bool:
        try:
            from executorch.backends.arm.ethosu import (  # noqa: F401
                EthosUCompileSpec,
                EthosUPartitioner,
            )
            import ethosu.vela  # noqa: F401 — the Vela compiler must be importable to lower
            return True
        except Exception:
            return False

    def quantizer(self):
        from executorch.backends.arm.quantizer import (
            EthosUQuantizer,
            get_symmetric_quantization_config,
        )
        q = EthosUQuantizer(_compile_spec())
        q.set_global(get_symmetric_quantization_config())
        return q

    def quantizes(self, quantize: bool = False) -> bool:
        # Integer-only NPU: every job is quantized, so pregen must always store a quantized
        # reference (the fp32 eager oracle is never the right comparison here).
        return True

    def lower(self, ep, example_inputs, quantize: bool = False):
        # Integer-only NPU: always PT2E-quantize first, regardless of the flag.
        ep = _quantize_pt2e(ep, example_inputs, self.quantizer())
        return self._lower(ep, example_inputs)

    def _lower(self, ep, example_inputs):
        from executorch.backends.arm.ethosu import EthosUPartitioner
        return lower_with_partitioner(ep, EthosUPartitioner(_compile_spec()))
