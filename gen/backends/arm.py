"""ARM/TOSA backend — lowers to the TOSA dialect for ARM accelerators (Ethos-U etc.).
Lowers on the host; runs only on the target device. Supports PT2E quantization
(TOSAQuantizer). Available only where the ARM partitioner imports.
"""

from __future__ import annotations

from .base import Backend, lower_with_partitioner


def _partitioner():
    # The ARM partitioner has moved across executorch versions; try the known locations.
    try:
        from executorch.backends.arm.tosa_partitioner import TOSAPartitioner
        return TOSAPartitioner()
    except Exception:
        from executorch.backends.arm.ethosu_partitioner import EthosUPartitioner
        return EthosUPartitioner()


class ArmBackend(Backend):
    name = "arm"
    runs_on_host = False
    supports_quantization = True

    def is_available(self) -> bool:
        try:
            _partitioner()
            return True
        except Exception:
            return False

    def quantizer(self):
        from executorch.backends.arm.quantizer import (
            TOSAQuantizer,
            get_symmetric_quantization_config,
        )
        q = TOSAQuantizer()
        q.set_global(get_symmetric_quantization_config())
        return q

    def _lower(self, ep, example_inputs):
        return lower_with_partitioner(ep, _partitioner())
