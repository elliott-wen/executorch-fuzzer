"""job.py — one graph: build it, write it out, run the PyTorch reference.

This is the whole per-graph unit of work. Nothing about who generates it enters into it: no
worker id, no process rank, no core count. That is what lets the parallel layer above be a
pure scheduling decision.

The token IS the job. It arrives from the parent, it is the graph's only source of randomness
— seeding the builder and, through it, the leaf data baked into the emitted source — and it
is the name the record is filed under. One identity, not a slot number and a seed that have
to be kept in step.

There is nothing to re-derive a graph from: Z3 hands back a different model depending on what
its global context has already solved, so a graph was never reproducible from its seed even
when one existed. The corpus stores the emitted source and every consumer reads it. The record
IS the graph.

The eager run is the oracle: what PyTorch says this graph computes. It is also the step that
can take the process down with it — a bad op can corrupt the heap rather than raise — which
is why `step` is written to be safely re-attempted and why the layer above expects to lose
processes. Everything before the eager call is cheap and pure; everything after it is a
write of an already-computed result.
"""

from __future__ import annotations

from collections.abc import Callable
import random

import torch

from mobile.generator.oracle import store
from mobile.generator.oracle.params import Params
from mobile.generator.graph import build_graph


def derive(ops, token: str, params: Params):
    """A token → a Graph, or None if none could be built.

    Retried because building is stochastic — a node is proposed, solved, and probed, and any
    of those can come up empty for reasons that a re-draw fixes. The retry count lives here,
    in the caller, rather than inside build_graph, so it can't silently multiply against a
    second retry loop somewhere else.
    """
    rng = random.Random(token)
    for _ in range(max(1, params.retries)):
        graph = build_graph(rng, ops, params.nodes, params.leaf_prob,
                            out_alias_prob=params.out_alias_prob)
        if graph is not None:
            return graph
    return None


def run_eager(src: str) -> tuple[list, list]:
    """Exec a graph source and run it, returning (inputs, outputs). Raises on a bad graph.

    The graph runs on CLONES of the leaves, and the leaves themselves are what's returned:
    a graph can mutate its own inputs (an `out=` buffer wired to a leaf), and the inputs the
    reference actually saw are the pre-mutation ones. Shipping anything else would compare
    two runs that started from different data.
    """
    ns: dict = {}
    exec(compile(src, "<corpus_graph>", "exec"), ns)
    g, leaves = ns["g"], ns["LEAVES"]
    with torch.no_grad():
        result = g(*[t.clone() for t in leaves])
    outputs = list(result) if isinstance(result, (tuple, list)) else [result]
    return leaves, outputs


def step(root, ops, token: str, params: Params,
         enter: Callable[[str], None]) -> tuple[str, str]:
    """Generate the graph `token` names and file it. Never raises; may hard-crash.

    Returns (outcome, detail) — the supervisor's vocabulary, reported verbatim:

        ready     a record was written
        gen_fail  no valid DAG came out of the builder
        emit      the graph exists but can't be written as source
        eager     PyTorch itself rejected the graph (invalid inputs, unsupported combo)

    An `eager` outcome is normal, not a defect: the generator proposes calls the constraints
    believe are valid, and PyTorch is the authority that says otherwise.

    `enter` is called with each stage before it starts. The eager run can take the process
    down rather than raise, and when it does that announcement is the only trace of where it
    was — see mobile.generator.supervisor.protocol.

    The record is written atomically and only after everything risky has already succeeded,
    so a token that never got that far simply has no record — which is what makes the corpus
    exactly the set of graphs that worked.
    """
    enter("generate")
    graph = derive(ops, token, params)
    if graph is None:
        return "gen_fail", f"no valid DAG in {params.retries} attempts"
    enter("emit")
    try:
        # The baked torch.manual_seed is the token's first 8 hex digits, so a generated
        # script carries the provenance of the token that produced it.
        src = graph.emit(seed=int(token[:8], 16), nonfinite_prob=params.nonfinite_prob)
    except Exception as e:                       # noqa: BLE001 — any emit failure is a skip
        return "emit", f"{type(e).__name__}: {e}"
    if src is None:
        return "emit", "un-emittable argument or missing schema"
    enter("eager")
    try:
        inputs, eager = run_eager(src)           # the oracle — and the step that can crash
    except Exception as e:                       # noqa: BLE001 — PyTorch rejecting the graph
        return "eager", f"{type(e).__name__}: {e}"
    enter("write")
    store.write_record(root, token, src, graph.describe(), inputs, eager)
    return "ready", ""
