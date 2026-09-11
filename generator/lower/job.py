"""job.py — one token: an oracle record in, a lowered .pte out.

The second half of generation. It reads what the oracle stage banked — the graph source and
the inputs the eager reference actually ran on — exports it, and lowers it for one backend.

The expensive, unrepeatable half of the work is already done: Z3 solving and graph building
happened once, and cannot be redone identically. So lowering never regenerates anything; it
re-execs the stored source purely to recover the callable, and takes the inputs from the
record rather than re-deriving them. That is what lets a second backend be lowered later, or
a failed lowering be retried after a fix, without disturbing the corpus.

Both risky calls are announced, because they fail for unrelated reasons and both can take
the process down rather than raise:

    export          torch.export — tracing the graph
    transformation  to_edge — ATen to Edge dialect, plus the edge passes
    lower           to_executorch — memory planning, delegation, serialization
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.export import export
from torch.export.graph_signature import OutputKind

from mobile.generator.lower import store
from mobile.generator.lower.backends import Backend, LowerStageError
from mobile.generator.oracle import store as oracle_store


class _Wrapped(torch.nn.Module):
    """torch.export needs an nn.Module; the corpus stores a plain function."""

    def __init__(self, g) -> None:
        super().__init__()
        self._g = g

    def forward(self, *args):
        return self._g(*args)


def callable_from(src: str):
    """Exec a stored graph source and hand back its `g`.

    Only the callable is taken. The inputs come from the record, because a graph may mutate
    its own leaves and the ones the reference saw are the pre-mutation originals — re-running
    `_make` here would produce different tensors and silently compare two different runs.
    """
    namespace: dict = {}
    exec(compile(src, "<corpus_graph>", "exec"), namespace)
    return namespace["g"]


def delegation(program) -> dict[str, int]:
    """How much of the graph the backend actually took.

    Counted off the lowered program rather than inferred: `delegated` is the real ops inside
    every delegate's captured subgraph (not the number of partitions), `portable` the ops
    still running as reference kernels, `calls` how many separate delegate blobs came out.
    A backend with no partitioner gives (0, N, 0).
    """
    import operator

    from executorch.exir.backend.utils import get_delegates, get_non_lowered_nodes
    from executorch.exir.lowered_backend_module import LoweredBackendModule

    graph_module = program.exported_program().graph_module
    portable = sum(1 for n in get_non_lowered_nodes(graph_module.graph)
                   if n.target is not operator.getitem)
    calls = len(get_delegates(graph_module.graph))
    delegated = 0
    for node in graph_module.graph.nodes:
        if node.op == "get_attr" and node.name.startswith("lowered_module_"):
            module = getattr(graph_module, node.name, None)
            if isinstance(module, LoweredBackendModule):
                sub = module.original_module.graph_module.graph
                delegated += sum(1 for x in sub.nodes
                                 if x.op == "call_function" and x.target is not operator.getitem)
    return {"delegated": delegated, "portable": portable, "calls": calls}


def user_output_positions(program) -> list[int]:
    """Which .pte output slots are the graph's REAL outputs.

    A lowered program returns the graph's outputs PLUS an alias for every mutated input — an
    `out=` buffer wired to a leaf shows up as an extra result. The eager reference has no
    such aliases, so the comparison has to know which positions to look at.
    """
    specs = program.exported_program().graph_signature.output_specs
    return [i for i, s in enumerate(specs) if s.kind == OutputKind.USER_OUTPUT]


def step(oracle_root, out_root, backend: Backend, token: str, quantize: bool,
         enter: Callable[[str], None]) -> tuple[str, str]:
    """Lower the graph `token` names. Never raises; may hard-crash.

    Returns (outcome, detail):

        ready           a .pte was written
        no_record       the oracle corpus has no such token (only if it was deleted or the
                        directory changed under us — the jobs come from a listing of it)
        export          torch.export refused the graph
        transformation  to_edge refused it
        lower           to_executorch refused it
        quant_ref       the quantized reference could not be produced

    A lowering failure is a finding, not just a skip: `transformation` and `lower` are the
    compiler declining a graph the runtime would happily run.
    """
    enter("read")
    record = oracle_store.read_record(oracle_root, token)
    if record is None:
        return "no_record", f"{token} is not in {oracle_root}"
    inputs = record["inputs"]

    enter("export")
    try:
        g = callable_from(record["src"])
        program = export(_Wrapped(g).eval(), tuple(t.clone() for t in inputs))
    except Exception as e:                          # noqa: BLE001 — export declining a graph
        return "export", f"{type(e).__name__}: {str(e)[:160]}"

    reference = None
    if backend.quantizes(quantize):
        enter("quant_ref")
        try:
            # A TUPLE, not the record's list: a backend may re-export here (nxp re-exports
            # strict, because its quantizer's conv/linear patterns only match that graph
            # shape), and torch.export rejects a list of example inputs outright.
            reference = backend.quantized_reference(program, tuple(t.clone() for t in inputs))
        except Exception as e:                      # noqa: BLE001
            return "quant_ref", f"{type(e).__name__}: {str(e)[:160]}"

    enter("lower")
    try:
        lowered = backend.lower(program, tuple(t.clone() for t in inputs), quantize=quantize)
    except LowerStageError as e:
        return e.stage, str(e)                      # "transformation" or "lower"
    except Exception as e:                          # noqa: BLE001
        return "lower", f"{type(e).__name__}: {str(e)[:160]}"

    enter("write")
    # The oracle record comes across too, so the lowering corpus is self-contained and the
    # executor needs one path rather than a join across two.
    store.copy_oracle(oracle_root, out_root, token)
    store.write_record(out_root, token, lowered.buffer, {
        "backend": backend.name,
        "quantized": backend.quantizes(quantize),
        "user_outputs": user_output_positions(lowered),
        "delegation": delegation(lowered),
    }, reference=reference)
    return "ready", ""
