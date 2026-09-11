"""mobile — ZeroMQ-brokered ExecuTorch differential fuzzing.

Generation happens offline, in two batch passes that write to disk, and a broker then
streams that work over ZeroMQ to executor clients (local runtimes, simulators, phones):

  # GENERATE (offline): graphs + their PyTorch reference once, then lower that same
  # oracle corpus for as many backends as you like. Both are resumable and supervised.
  python -m mobile.generator.oracle tmp/oracle --count 100000
  python -m mobile.generator.lower  tmp/oracle tmp/pte --backend portable

  # BROKER (one): routes jobs to executors, tallies, saves repros.
  python -m mobile broker [--job-port 15554] [--client-port 15555] [--ctrl-port 15556]

  # FEED (one): join the two corpora by token, stream jobs, diff the results.
  python -m mobile feed --pte tmp/pte --oracle tmp/oracle

  # CLIENT (one or more / per device): pull jobs, run .pte. Each backend has its OWN
  # self-contained executor folder (the host analog of the Android/FVP/QNN clients);
  # the in-process ExecuTorch runtime runs both xnnpack- and portable-lowered programs:
  #   python local_client/xnnpack_client/xnnpack_client.py --host <broker-ip> [--job-timeout 30]

Start the broker first, then feed + any number of clients (locally or on more
machines/phones). mobile/local_fleet.py manages the feeder + client fleet with respawn.
"""

from __future__ import annotations

import os
# CPU-only everywhere; before any torch import in the spawned role.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(prog="mobile", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    br = sub.add_parser("broker", help="pure load-balancing router (feeders ↔ workers)")
    br.add_argument("--host", default="0.0.0.0", help="(informational; broker binds all)")
    br.add_argument("--job-port", type=int, default=15554, help="frontend: feeders connect here")
    br.add_argument("--client-port", type=int, default=15555, help="backend: workers connect here")
    br.add_argument("--ctrl-port", type=int, default=15556)
    br.add_argument("--heartbeat", type=float, default=3.0,
                    help="seconds between broker status lines (with -v)")
    br.add_argument("-v", "--verbose", action="store_true")

    fd = sub.add_parser("feed", help="stream jobs to the broker, diff results, tally (a dir OR one graph)")
    fd.add_argument("jobs", nargs="*", help="specific job_id(s) or .job path(s) to replay (→ rich JSON); "
                                            "omit to feed everything that lowered (→ TSV)")
    fd.add_argument("--host", default="127.0.0.1", help="broker host")
    fd.add_argument("--job-port", type=int, default=15554)
    fd.add_argument("--ctrl-port", type=int, default=15556)
    fd.add_argument("--pte", default="tmp/pte", help="lowering corpus: the .pte to run")
    fd.add_argument("--oracle", default="tmp/oracle", help="oracle corpus: inputs + reference")
    fd.add_argument("--from-tsv", default=None, help="also take job_ids from a results TSV (col 2)")
    fd.add_argument("--status", default=None, help="with --from-tsv: keep only rows of this status")
    fd.add_argument("-n", "--graphs", type=int, default=0,
                    help="max jobs to feed; 0 = everything (also caps --from-tsv)")
    fd.add_argument("--timeout", type=float, default=30.0,
                    help="per-job timeout (a worker that never answers → TIMEOUT)")
    fd.add_argument("--window", type=int, default=64,
                    help="max in-flight jobs (end-to-end backpressure)")
    fd.add_argument("--skip-log", default="tmp/skip_reasons_mobile.tsv",
                    help="failing rows: status + token + reason + op-chain (the graph is in the oracle corpus)")
    fd.add_argument("--heartbeat", type=float, default=3.0,
                    help="seconds between feed status lines")
    fd.add_argument("-v", "--verbose", action="store_true")

    opts = ap.parse_args()

    if opts.cmd == "broker":
        from mobile.executor.broker import run_broker
        return run_broker(opts.job_port, opts.client_port, opts.ctrl_port,
                          opts.heartbeat, opts.verbose)
    if opts.cmd == "feed":
        from mobile.executor.feed import run_feeder
        return run_feeder(opts.host, opts.job_port, opts.ctrl_port, opts.pte, opts.oracle,
                          opts.jobs, opts.from_tsv, opts.status, opts.graphs,
                          opts.timeout, opts.skip_log, opts.window,
                          opts.heartbeat, opts.verbose)
    raise SystemExit(f"unknown command: {opts.cmd}")  # argparse(required=True) makes this unreachable


if __name__ == "__main__":
    sys.exit(main())
