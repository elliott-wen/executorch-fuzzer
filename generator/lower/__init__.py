"""lower — oracle records to ExecuTorch programs, in bulk, in parallel.

The second half of generation, and a separate batch pass on purpose: graph building is
expensive and unrepeatable (Z3 gives a different model each time), so it is banked once by
the oracle stage and lowering reads it. That way a second backend, or a retry after fixing a
compiler bug, costs only the lowering.

    from mobile.generator.lower import lower, summarize

    lower("tmp/oracle", "tmp/pte/portable", backend="portable", count=100_000, workers=64)
    print(summarize("tmp/pte/portable"))

or from the command line:

    python -m generator.lower tmp/oracle tmp/pte/portable --count 100000 --workers 64

A compiler refusal is a finding, not merely a skip: `to_edge` and `to_executorch` outcomes
are the compiler declining a graph the runtime would have run. Sub-stages are named after
the API call that refused, so `lower` only ever names this stage.

  backends/  the lowering targets, probed lazily one at a time
  job        one token: read the record, export, lower, write the .pte
  store      the on-disk layout for lowered programs
  worker     the supervised process — one token per job
  fleet      what a worker is, what the jobs are
"""

from __future__ import annotations

from mobile.generator.lower.fleet import lower, summarize

__all__ = ["lower", "summarize"]

# `job`, `store` and `backends` are deliberately NOT imported here: they pull in torch and an
# SDK, and importing this package must stay cheap for the supervising process, which builds
# worker command lines but never lowers anything. Workers import them directly.
