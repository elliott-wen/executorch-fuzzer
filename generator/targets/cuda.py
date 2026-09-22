"""cuda — the ExecuTorch CUDA (AOTI) backend, on the portable floor.

Portable's rules, unchanged, and nothing else yet — the same starting point as the other
delegate targets: whatever the partitioner declines runs a portable kernel that still needs
portable's contracts.

TWO THINGS SET THIS TARGET APART, and both should shape anything added here.

  1. It is NOT a mobile target. Every other backend in this tree is an edge runtime with a
     narrow dtype vocabulary — which is exactly why five of them wanted a dtype rule. CUDA
     via AOTInductor compiles real PyTorch kernels for an H200, so the dtype story that
     dominated samsung/coreml/openvino/mediatek almost certainly does NOT apply. Expect the
     losses to look like vulkan's (toolchain and compile failures) rather than samsung's.

  2. QuantMode.NEVER — float only, so the comparison is against the fp32 eager oracle, with
     none of the quantized_reference machinery that cadence, ethos-u and nxp need. That also
     means the PT2E observer gaps which sank nxp (histogram_cpu has no integer kernel) cannot
     appear here.

RUNS IN A DIFFERENT VENV, AND ON A GPU. The main .venv is CPU-only on purpose (see
requirements.txt: "Do NOT install a +cuXXX torch here"), so CUDA lives in .venv-cuda with
torch 2.12.0+cu130, and lowering workers must be started with `--python .venv-cuda/bin/python`.
The parent process is unaffected, since it never imports torch.

Worth watching, because it has no analogue on the edge backends: AOTInductor COMPILES per
graph, so lowering is expected to be slow and to hold GPU memory. The box has three H200s
shared with other users — size the worker count against free memory rather than core count.
"""

from __future__ import annotations

from typing import Any

from mobile.generator.targets import portable


def axioms(op_name: str, params: list[tuple[str, str]], variables: dict[str, Any]) -> list:
    """Portable's rules, unchanged. Nothing CUDA-specific yet — see the module docstring."""
    return portable.axioms(op_name, params, variables)
