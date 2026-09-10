"""worker.py — the supervised process that generates graphs.

One job is one token, read from stdin; the answer is one outcome line on stdout. The
protocol, the flushing, and the crash reporting all belong to mobile.generator.supervisor —
this file is only the work.

The process is long-lived and makes many graphs, so the op catalog is loaded once. That is
safe despite the eager run being able to abort the process, because the supervisor dispatches
one token at a time and waits for its answer: whatever is in flight when this process dies is
exactly one graph, and the parent already knows which token it was.
"""

from __future__ import annotations

import os

# Set before torch is imported: with one worker per core, each must stay single-threaded, or
# every one spawns a full OpenMP pool and the box spends its time in the scheduler.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from mobile.generator.oracle.job import step  # noqa: E402
from mobile.generator.oracle.params import Params  # noqa: E402
from mobile.generator.graph import load_ops  # noqa: E402
from mobile.generator.supervisor import serve  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="generate the graphs whose tokens arrive on stdin")
    parser.add_argument("--out", required=True)
    Params.add_arguments(parser)
    args = parser.parse_args(argv)
    params = Params.from_args(args)

    # ~3.8s, paid once for the many indices this process will handle — which is the whole
    # reason the supervisor keeps workers alive rather than spawning one per graph.
    ops = load_ops(composable=params.composable, verbose=False)

    def handle(job: str, enter) -> tuple[str, str]:
        # The job IS the token: a fresh uuid4 from the parent, so it cannot collide with a
        # record already on disk and there is nothing to check for first.
        return step(args.out, ops, job, params, enter)

    serve(handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
