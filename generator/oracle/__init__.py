"""oracle — generate graphs and their PyTorch reference, in bulk, in parallel.

The stage above `generator.graph`: turn compute into a directory of runnable graphs
each paired with what PyTorch says it computes. It is backend-agnostic on purpose — the
eager reference is the same no matter what you later lower to, so it is produced once here
and every backend's export stage reads it.

    from mobile.generator.oracle import Params, generate, summarize

    generate("tmp/oracle", params=Params(nodes=8), count=1_000_000, workers=48)
    print(summarize("tmp/oracle"))

or from the command line:

    python -m mobile.generator.oracle tmp/oracle --count 1000000 --workers 48
    python -m mobile.generator.oracle tmp/oracle --stats

A graph's identity is one random token, minted by the parent. It is the job the supervisor
dispatches, the only randomness the graph gets, and the name the record is filed under —
`<token[:3]>/<token[3:]>.oracle`, sharded the way git shards objects. One identity, so there
is no index space to keep in step with what is on disk.

Nothing is reproducible from that token, and nothing needs to be: Z3 returns a different
satisfying model depending on what its global context has already solved (six identical
solves, six different models), so a graph was never re-derivable even when a seed existed.
The corpus is simply the set of records in it, and every consumer reads the stored `.py`.
The record IS the graph. Generation is therefore append-only — running again makes more
graphs rather than filling gaps, and there is nothing to resume.

Crash isolation is the operating system's, arranged by mobile.generator.supervisor: graphs run
only in worker processes, jobs are dispatched one at a time and acknowledged, so a graph that
aborts PyTorch costs exactly the one that was in flight — and the worker says which stage it
was in when it went.

  params   how graphs are shaped; torch-free, so the supervisor can read it
  job      one token: derive → emit → run eager. The unit of work.
  store    the on-disk layout: records on disk.
  worker   the supervised process — one token per job, driven by generator.supervisor
  fleet    what a worker is, what the jobs are, what to do with each answer
"""

from __future__ import annotations

from mobile.generator.oracle.fleet import generate, summarize
from mobile.generator.oracle.params import Params

__all__ = ["Params", "generate", "summarize"]

# `oracle` and `store` are deliberately NOT imported here. They pull in torch, and importing
# this package must stay cheap for the supervising process, which builds worker command lines
# but never runs a graph itself. Workers import them directly.
