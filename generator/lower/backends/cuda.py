"""cuda — NVIDIA GPU via ExecuTorch's AOTInductor (AOTI) path.

Unlike the mobile delegates this is an AOTInductor backend: to_edge_transform_and_lower does
not merely tag a subgraph, it invokes AOTInductor, which COMPILES the partition into a CUDA
shared object during lowering (needing CUDA PyTorch, triton and nvcc) and embeds that .so in
the .pte as a named-data blob. At run time `aoti_cuda_backend` loads the blob. Two
consequences:

  * Lowering must happen on a box with a GPU and a CUDA toolchain — a CUDA corpus cannot be
    produced on a CPU-only worker. Lowering and execution are co-located on the GPU host, and
    the whole lane runs in .venv-cuda (CUDA torch, executorch built with
    EXECUTORCH_BUILD_CUDA=ON), never the CPU .venv. Point the fleet at it with --python.
  * runs_on_host is True because "host" here IS the GPU server: the .pte executes in-process
    on the ExecuTorch runtime, with no external simulator.

The partitioner is AOTI whole-graph — a single partition of everything AOTInductor handles,
with the remainder falling back to portable — so one unsupported op tends to fail lowering
wholesale, the same all-or-nothing yield profile as samsung and mediatek.

QuantMode.NEVER for now. The backend does support INT4 *weight* quantization, but through a
distinct AOTI quant flow rather than a drop-in PT2E quantizer, so that is a later dimension.
"""

from __future__ import annotations

from mobile.generator.lower.backends.base import Backend, QuantMode, lower_with_partitioner

#: the exported method AOTInductor compiles. The corpus exports a single-method module, so
#: this is always "forward".
METHOD_NAME = "forward"


class CudaBackend(Backend):
    name = "cuda"
    runs_on_host = True               # host == the GPU box; runs in-process on the ET runtime
    quant = QuantMode.NEVER

    def is_available(self) -> bool:
        try:
            import torch
            import triton  # noqa: F401 — AOTInductor CUDA codegen needs triton
            from executorch.backends.cuda.cuda_backend import CudaBackend  # noqa: F401
            from executorch.backends.cuda.cuda_partitioner import (  # noqa: F401
                CudaPartitioner,
            )
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _lower(self, ep, example_inputs):
        from executorch.backends.cuda.cuda_backend import CudaBackend as _EtCudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

        specs = [_EtCudaBackend.generate_method_name_compile_spec(METHOD_NAME)]
        return lower_with_partitioner(ep, CudaPartitioner(specs))
