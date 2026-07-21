"""export_et.py — HOST side: functional graph source → eager reference + .pte bytes.

Given a graph-definition source string (from generate.graph_ir.GenGraph.emit), this:
  1. execs the source to recover `g` and `LEAVES` (the concrete input tensors),
  2. runs the eager reference `g(*LEAVES)` (the differential ORACLE),
  3. lowers `g` to an ExecuTorch program via torch.export → to_edge → to_executorch,
     returning the `.pte` bytes.

The inputs the eager reference ran on are the *same tensors* shipped to the client,
so the diff sees byte-identical inputs (see plan: determinism). RNG-laced graphs are
excluded upstream (runnable_ops); export/eager failures are reported as SKIP reasons.

This module is host-only (needs torch + executorch.exir). The client side
(et_runner.py) needs only the ExecuTorch runtime.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

warnings.filterwarnings("ignore")

import torch  # noqa: E402
from torch.export import export  # noqa: E402
from torch.export.graph_signature import OutputKind  # noqa: E402

# Per-backend lowering targets. Each backend owns its own lowering and quantization under
# mobile/gen/export/backends/. This module stays target-agnostic: it captures the eager oracle + the
# exported program, then hands off to the chosen backend (optionally with quantization).
from .backends import (  # noqa: E402,F401  (re-exported API)
    available_backends,
    get_backend,
    backend_supports_quantization,
    quantizable_backends,
    LowerStageError,
)


@dataclass
class ExportResult:
    status: str                 # "READY" | "SKIP"
    detail: str = ""
    stage: str = "ready"        # pipeline stage this result belongs to — one of:
                                # config|emit|eager|export|transformation|lower|quant_ref|ready.
                                # For SKIP it names WHERE the graph fell out; pregen buckets on it.
    pte: bytes | None = None
    inputs: list | None = None  # the concrete leaf tensors (host-materialized)
    eager: list | None = None   # eager reference outputs (tuple flattened to list)
    user_pos: list | None = None  # which .pte output indices are real graph outputs
                                  # (USER_OUTPUT) — the rest are mutated-input aliases
    delegated_ops: int = 0      # aten ops absorbed into a backend delegate (0 for portable)
    non_delegated_ops: int = 0  # ops left running as portable/reference kernels
    delegate_calls: int = 0     # number of delegate partitions (executorch_call_delegate)


def _flatten_outputs(r) -> list:
    if isinstance(r, (tuple, list)):
        return list(r)
    return [r]


def _delegation_counts(exe) -> tuple[int, int, int]:
    """(delegated_ops, non_delegated_ops, delegate_calls) for a lowered program.

    `delegated_ops` counts the real ops inside every delegate's captured subgraph
    (not the number of partitions); `non_delegated_ops` counts the ops still lowered
    to portable/reference kernels; `delegate_calls` is how many separate delegate
    blobs the partitioner produced. Portable (no partitioner) ⇒ (0, N, 0)."""
    from executorch.exir.backend.utils import get_delegates, get_non_lowered_nodes
    from executorch.exir.lowered_backend_module import LoweredBackendModule
    import operator

    gm = exe.exported_program().graph_module
    non_delegated = sum(1 for n in get_non_lowered_nodes(gm.graph)
                        if n.target is not operator.getitem)
    delegate_calls = len(get_delegates(gm.graph))
    delegated = 0
    for node in gm.graph.nodes:
        if node.op == "get_attr" and node.name.startswith("lowered_module_"):
            lm = getattr(gm, node.name, None)
            if isinstance(lm, LoweredBackendModule):
                sub = lm.original_module.graph_module.graph
                delegated += sum(1 for x in sub.nodes
                                 if x.op == "call_function"
                                 and x.target is not operator.getitem)
    return delegated, non_delegated, delegate_calls


class _GraphModule(torch.nn.Module):
    """torch.export requires an nn.Module; wrap the plain functional `g`."""

    def __init__(self, g):
        super().__init__()
        self._g = g

    def forward(self, *args):
        return self._g(*args)


def load_functional_source(src: str) -> tuple[Any, list]:
    """exec a functional graph source string → (g callable, LEAVES list)."""
    ns: dict = {}
    exec(compile(src, "<mobile_graph>", "exec"), ns)
    return ns["g"], ns["LEAVES"]


def build_job(src: str, backend: str = "portable", quantize: bool = False) -> ExportResult:
    """Functional source → ExportResult(READY with pte+inputs+eager, or SKIP).

    `backend` selects the lowering target: "portable" (CPU kernels) or a delegate
    (xnnpack/vulkan/...) via available_backends(). `quantize` toggles PT2E quantization
    for backends that support it (the 2nd measurement dimension). The eager oracle is
    unchanged — only the .pte differs — so a backend run is still diffed against host eager.

    Never raises: any failure becomes a SKIP with a reason string, so the server
    loop can log it and move on."""
    backend_obj = get_backend(backend)
    if backend_obj is None:
        return ExportResult("SKIP", f"unknown backend {backend!r} "
                                    f"(have: {', '.join(available_backends())})", stage="config")
    if quantize and not backend_obj.supports_quantization:
        return ExportResult("SKIP", f"quantize unsupported by backend {backend!r}", stage="config")
    try:
        g, leaves = load_functional_source(src)
    except Exception as e:
        return ExportResult("SKIP", f"emit/exec {type(e).__name__}: {e}", stage="emit")

    # Eager reference — the oracle. A clean raise here means the graph is an
    # invalid-input graph (out-of-bounds index, etc.) → SKIP, not a bug.
    try:
        with torch.no_grad():
            eager = _flatten_outputs(g(*[t.clone() for t in leaves]))
    except Exception as e:
        return ExportResult("SKIP", f"eager {type(e).__name__}: {e}", stage="eager")

    # Lower to ExecuTorch.
    try:
        ep = export(_GraphModule(g).eval(), tuple(t.clone() for t in leaves))
    except Exception as e:
        return ExportResult("SKIP", f"export {type(e).__name__}: {str(e)[:160]}", stage="export")
    try:
        exe = backend_obj.lower(ep, tuple(t.clone() for t in leaves), quantize=quantize)
        pte = exe.buffer
        # The .pte returns the graph's real outputs PLUS aliases of any mutated input
        # (e.g. an out=/indices= buffer wired to a leaf). Record which output positions
        # are the REAL graph outputs (USER_OUTPUT) so the client compares only those —
        # the eager reference never includes the mutated-input aliases.
        specs = exe.exported_program().graph_signature.output_specs
        user_pos = [i for i, s in enumerate(specs) if s.kind == OutputKind.USER_OUTPUT]
        deleg_ops, nondeleg_ops, deleg_calls = _delegation_counts(exe)
    except LowerStageError as e:
        # Split lowering: "transformation" (to_edge) vs "lower" (to_executorch/delegate).
        return ExportResult("SKIP", f"{e.stage} {e}", stage=e.stage)
    except Exception as e:
        return ExportResult("SKIP", f"to_executorch {type(e).__name__}: {str(e)[:160]}", stage="lower")

    # For a QUANTIZED job, the fp32 eager oracle is the wrong reference — an int8 backend
    # diverges from it by quantization error alone. Replace it with the quantized reference
    # (the PT2E-converted graph run on CPU, same quantization space as the device), so a
    # surviving diff is a real backend/compiler bug. See backends/base.quantized_reference.
    if backend_obj.quantizes(quantize):
        try:
            eager = backend_obj.quantized_reference(ep, tuple(t.clone() for t in leaves))
        except Exception as e:
            return ExportResult("SKIP", f"quant-ref {type(e).__name__}: {str(e)[:160]}", stage="quant_ref")

    return ExportResult("READY", "", pte=bytes(pte), inputs=leaves, eager=eager,
                        user_pos=user_pos, delegated_ops=deleg_ops,
                        non_delegated_ops=nondeleg_ops, delegate_calls=deleg_calls)
