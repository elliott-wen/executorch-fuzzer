"""worker.py — the supervised process that lowers oracle records to .pte.

One job is one token, read from stdin; the answer is one outcome line on stdout. The
protocol, the flushing and the crash reporting belong to mobile.generator.supervisor — this
file is only the work.

The process is long-lived and lowers many graphs, so the backend is probed once. Lowering
accumulates global state in a way that is hard to see (an exir verifier that grew a
module-level list on every call once slowed a whole run to a crawl), which is exactly what
the supervisor's worker recycling is for — see fleet.MAX_GRAPHS_PER_WORKER.
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

from mobile.generator.lower import backends, store  # noqa: E402
from mobile.generator.lower.job import step  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="lower the graphs whose tokens arrive on stdin")
    parser.add_argument("--oracle", required=True, help="oracle corpus to read")
    parser.add_argument("--out", required=True, help="where to write .pte records")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args(argv)

    backend = backends.get(args.backend)
    if backend is None:
        print(f"backend {args.backend!r} is not available in this install", file=sys.stderr)
        return 2

    def handle(job: str, enter) -> tuple[str, str]:
        if store.exists(args.out, job):
            return "done", ""                 # this token is already lowered
        return step(args.oracle, args.out, backend, job, args.quantize, enter)

    from mobile.generator.supervisor import serve
    serve(handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
