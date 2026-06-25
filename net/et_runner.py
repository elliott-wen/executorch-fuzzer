"""et_runner.py — CLIENT side: run one .pte on the ExecuTorch runtime.

Thin and torch-light (it does import torch for tensor I/O, but does NO generation,
no z3, no export). This is the exact work a mobile phone client mirrors natively:
load the program, execute the forward method on the provided inputs, return outputs.

`run_pte(pte_bytes, inputs)` returns a list of output tensors. A hard kernel failure
in the ExecuTorch runtime is a NATIVE abort that takes the process down — which is
why the client runs each job in a respawnable subprocess and the server treats a
dropped connection as CRASH (see client.py / server.py).
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")


def run_pte(pte_bytes: bytes, inputs: list) -> list:
    """Load `pte_bytes` into the ExecuTorch runtime and run forward(*inputs).
    Returns a list of output tensors. May raise (Python) or hard-abort (native)."""
    from executorch.runtime import Runtime

    rt = Runtime.get()
    program = rt.load_program(pte_bytes)
    method = program.load_method("forward")
    out = method.execute([t.clone() for t in inputs])
    return list(out)
