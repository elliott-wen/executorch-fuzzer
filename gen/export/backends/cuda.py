"""CUDA backend — NVIDIA GPU delegate via ExecuTorch's AOTInductor (AOTI) path.

Unlike the mobile delegates (xnnpack/openvino/qnn/...) this is an *AOTInductor* backend:
`to_edge_transform_and_lower` doesn't just tag a subgraph, it invokes AOTInductor which
COMPILES the partition into a CUDA shared object right here (needs a CUDA PyTorch + triton +
nvcc), then embeds that .so as a named-data blob in the .pte. At runtime ExecuTorch's
`aoti_cuda_backend` loads the blob and runs it on the GPU. Two consequences:

  * Lowering must happen on a box with a GPU + CUDA toolchain — you CANNOT pregen a CUDA
    corpus on a CPU-only worker (the compile step runs device codegen). So pregen and the
    executor client are co-located on the GPU host, and this whole lane runs in .venv-cuda
    (CUDA torch + executorch built with EXECUTORCH_BUILD_CUDA=ON), never the CPU .venv.
  * `runs_on_host = True`: "host" here IS the GPU server, so the .pte executes in-process on
    the ExecuTorch runtime exactly like the openvino/xnnpack clients — no external simulator.

The partitioner is AOTI whole-graph (single partition of everything AOTInductor can handle;
the rest falls back to portable), so a graph with one unsupported op tends to fail lowering
wholesale — the same all-or-nothing yield profile as samsung/mediatek.

QuantMode.NEVER for now: the backend supports INT4 *weight* quantization, but that's a
distinct AOTI quant flow (not a drop-in PT2E quantizer), so it's a later dimension — v1 is
float only.

Available only in an install where the CUDA backend imports AND a GPU is visible
(`torch.cuda.is_available()`); i.e. .venv-cuda on this box, not the CPU .venv.
"""

from __future__ import annotations

from .base import Backend, QuantMode, lower_with_partitioner

# The exported method AOTInductor compiles for. build_job exports a single-method module
# (_GraphModule.forward), so this is always "forward".
METHOD_NAME = "forward"


class CudaBackend(Backend):
    name = "cuda"
    runs_on_host = True               # host == the GPU box; runs in-process on the ET runtime
    quant = QuantMode.NEVER

    def is_available(self) -> bool:
        try:
            import torch
            from executorch.backends.cuda.cuda_backend import (  # noqa: F401
                CudaBackend as _EtCudaBackend,
            )
            from executorch.backends.cuda.cuda_partitioner import (  # noqa: F401
                CudaPartitioner,
            )
            import triton  # noqa: F401 — AOTInductor CUDA codegen needs triton
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _lower(self, ep, example_inputs):
        from executorch.backends.cuda.cuda_backend import (
            CudaBackend as _EtCudaBackend,
        )
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

        specs = [_EtCudaBackend.generate_method_name_compile_spec(METHOD_NAME)]
        return lower_with_partitioner(ep, CudaPartitioner(specs))
