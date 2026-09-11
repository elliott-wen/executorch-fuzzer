"""CLI: python -m mobile.generator.oracle <dir> [--count N] [--workers N]

    # a million graphs across every core, resumable — re-run the same line to continue
    python -m mobile.generator.oracle tmp/oracle --count 1000000 --workers 48

    # what came out
    python -m mobile.generator.oracle tmp/oracle --stats

    # one graph, as source
    python -m mobile.generator.oracle tmp/oracle --show 4032
"""

from __future__ import annotations

import argparse
import sys

from mobile.generator.oracle.fleet import generate, summarize
from mobile.generator.oracle.params import Params


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mobile.generator.oracle",
                                 description="generate graphs + PyTorch oracle in parallel")
    ap.add_argument("out", help="corpus directory (created if absent, added to if present)")
    ap.add_argument("--count", type=int, default=10_000, help="how many graphs (default 10000)")
    ap.add_argument("--workers", type=int, default=0,
                    help="worker processes; 0 = one per core (default)")
    Params.add_arguments(ap)
    ap.add_argument("--stats", action="store_true", help="report on the corpus and exit")
    ap.add_argument("--show", metavar="TOKEN",
                    help="print one graph's stored source and exit")
    a = ap.parse_args(argv)

    if a.stats:
        print(summarize(a.out))
        return 0

    params = Params.from_args(a)

    if a.show is not None:
        from mobile.generator.oracle import store   # pulls in torch; only this path needs it

        record = store.read_record(a.out, a.show)
        if record is not None:
            print(f"# {record['desc']}\n{record['src']}")
            return 0
        source = store.read_source(a.out, a.show)   # a graph the eager run never survived
        if source is None:
            print(f"# {a.show} is not in this corpus")
            return 1
        print(f"# no oracle — this graph failed or crashed the eager run\n{source}")
        return 0

    generate(a.out, params=params, count=a.count, workers=a.workers)
    print()
    print(summarize(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
