#!/usr/bin/env python3
"""feed.py — stream a corpus to the broker once, diff results vs eager, tally.

Run it straight from the repo root, no -m / PYTHONPATH needed:

    python feed.py --corpus corpus/xnnpack --host <broker-ip>

The broker dispatches each job to whatever workers are attached (local clients or a remote
phone). The feeder keeps the eager reference, diffs returned outputs, and writes the skip-log.
Pass specific job_id(s)/.job path(s) as positional args to replay just those (rich JSON).
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# Put the repo's PARENT on sys.path so `import mobile` resolves wherever this is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mobile.net.feed import run_feeder


def main() -> int:
    ap = argparse.ArgumentParser(description="stream jobs to the broker, diff results, tally")
    ap.add_argument("jobs", nargs="*", help="specific job_id(s) or .job path(s) to replay "
                                            "(→ rich JSON); omit to feed the whole --corpus dir")
    ap.add_argument("--host", default="127.0.0.1", help="broker host")
    ap.add_argument("--job-port", type=int, default=15554)
    ap.add_argument("--ctrl-port", type=int, default=15556)
    ap.add_argument("--corpus", default="tmp/corpus", help="corpus dir (also resolves bare job_ids)")
    ap.add_argument("--from-tsv", default=None, help="also take job_ids from a results TSV (col 2)")
    ap.add_argument("--status", default=None, help="with --from-tsv: keep only rows of this status")
    ap.add_argument("-n", "--graphs", type=int, default=0,
                    help="max jobs to feed; 0 = the whole corpus (also caps --from-tsv)")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="per-job timeout (a worker that never answers → TIMEOUT)")
    ap.add_argument("--window", type=int, default=64,
                    help="max in-flight jobs (end-to-end backpressure)")
    ap.add_argument("--skip-log", default="tmp/skip_reasons_mobile.tsv",
                    help="failing rows: status + job_id + reason + op-chain")
    ap.add_argument("--heartbeat", type=float, default=3.0, help="seconds between feed status lines")
    ap.add_argument("-v", "--verbose", action="store_true")
    opts = ap.parse_args()
    return run_feeder(opts.host, opts.job_port, opts.ctrl_port, opts.corpus,
                      opts.jobs, opts.from_tsv, opts.status, opts.graphs,
                      opts.timeout, opts.skip_log, opts.window,
                      opts.heartbeat, opts.verbose)


if __name__ == "__main__":
    sys.exit(main())
