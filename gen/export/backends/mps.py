"""MPS backend (Apple) — delegates to the Metal Performance Shaders GPU. Available only on
macOS. Runs on the Mac host. fp16/fp32 path (no PT2E quantizer).

The MPS partitioner needs compile specs (MPSCompileSpec). Validated on macOS; on Linux this
backend isn't available and is omitted from the registry.
"""

from __future__ import annotations

from .base import Backend, QuantMode, lower_with_partitioner


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
        from executorch.backends.apple.mps.mps_preprocess import MPSBackend
        from executorch.exir.backend.compile_spec_schema import CompileSpec  # noqa: F401
        from executorch.backends.apple.mps.utils.mps_utils import MPSCompileSpec

        # fp16 on the GPU; mirrors the other delegates' default precision.
        return MPSCompileSpec(use_fp16=True).gen_compile_spec()

    def _lower(self, ep, example_inputs):
        from executorch.backends.apple.mps.partition.mps_partitioner import MPSPartitioner

        return lower_with_partitioner(ep, MPSPartitioner(compile_specs=self._compile_specs()))
