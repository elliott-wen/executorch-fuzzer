"""mps (Apple) — delegates to the Metal Performance Shaders GPU.

Available only on macOS, where it also runs. fp16/fp32; no PT2E quantizer. On Linux the
partitioner does not import, so the backend reports itself unavailable and is never used.
"""

from __future__ import annotations

from mobile.generator.lower.backends.base import Backend, QuantMode, lower_with_partitioner


class MpsBackend(Backend):
    name = "mps"
    runs_on_host = True               # MPS executes on the Mac host
    quant = QuantMode.NEVER

    def is_available(self) -> bool:
        try:
            from executorch.backends.apple.mps.partition.mps_partitioner import (  # noqa: F401
                MPSPartitioner,
            )
            return True
        except Exception:
            return False

    def _compile_specs(self):
        from executorch.backends.apple.mps.utils.mps_utils import MPSCompileSpec

        # fp16 on the GPU; mirrors the other delegates' default precision.
        return MPSCompileSpec(use_fp16=True).gen_compile_spec()

    def _lower(self, ep, example_inputs):
        from executorch.backends.apple.mps.partition.mps_partitioner import MPSPartitioner

        return lower_with_partitioner(ep, MPSPartitioner(compile_specs=self._compile_specs()))
