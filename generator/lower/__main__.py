"""CLI: python -m mobile.generator.lower <oracle-dir> <out-dir> [--backend portable]

    # lower a whole oracle corpus for one backend
    python -m mobile.generator.lower corpus/oracle_100k corpus/pte_portable \
        --backend portable --workers 64

    # what came out
    python -m mobile.generator.lower corpus/oracle_100k corpus/pte_portable --stats
"""

from __future__ import annotations

import argparse
import sys

from mobile.generator.lower.fleet import lower, summarize


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mobile.generator.lower",
                                 description="lower an oracle corpus to .pte, in parallel")
    ap.add_argument("oracle", help="oracle corpus to read")
    ap.add_argument("out", help="directory for .pte records (one per backend)")
    ap.add_argument("--backend", default="portable",
                    help="lowering target (default: portable)")
    ap.add_argument("--quantize", action="store_true",
                    help="PT2E-quantize first, on backends that support it")
    ap.add_argument("--count", type=int, default=None,
                    help="how many graphs to lower (default: the whole corpus)")
    ap.add_argument("--workers", type=int, default=0, help="0 = one per core (default)")
    ap.add_argument("--python", default=None,
                    help="interpreter for the workers, for backends with their own venv")
    ap.add_argument("--list", action="store_true", help="list usable backends and exit")
    ap.add_argument("--stats", action="store_true", help="report on `out` and exit")
    a = ap.parse_args(argv)

    if a.list:
        from mobile.generator.lower import backends
        print("available:", ", ".join(backends.available()))
        return 0
    if a.stats:
        print(summarize(a.out))
        return 0

    lower(a.oracle, a.out, backend=a.backend, count=a.count, workers=a.workers,
          quantize=a.quantize, python=a.python)
    print()
    print(summarize(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
