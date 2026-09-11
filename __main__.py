"""mobile — ZeroMQ-brokered ExecuTorch differential fuzzing.

Generation is PRE-GENERATED to disk (reproducible, every graph inspectable), then a
broker streams that corpus over ZeroMQ to executor clients (local now, phones later):

  # PREGEN (offline): a graph is a pure function of (seed, index) — generate the shared,
  # backend-agnostic oracle once, then export it for as many backends as you want.
  python -m mobile pregen --mode oracle --out tmp/oracle --count 10000
  python -m mobile pregen --mode export --oracle tmp/oracle --out tmp/corpus --backend portable
      (parallelize either with mobile/pregen_fleet.py)

  # BROKER (one): routes jobs to executors, tallies, saves repros.
  python -m mobile broker [--job-port 15554] [--client-port 15555] [--ctrl-port 15556]

  # FEED (one): stream the corpus to the broker (PUSH) once, then exit.
  python -m mobile feed --corpus tmp/corpus

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

    pg = sub.add_parser("pregen", help="pre-generate a corpus of jobs to disk (offline)")
    pg.add_argument("--mode", default="full", choices=("oracle", "export", "full"),
                    help="oracle: graph gen+eager only (shared, backend-agnostic). "
                         "export: torch.export+lower an existing oracle for one backend. "
                         "full: oracle+export in one process (default)")
    pg.add_argument("--out", default="tmp/corpus", help="corpus directory to write")
    pg.add_argument("--oracle", default="", help="oracle directory to read (--mode export)")
    pg.add_argument("--count", type=int, default=10000, help="how many indices to attempt, "
                    "starting at --start")
    pg.add_argument("--nodes", type=int, default=8, help="target real-op nodes per DAG "
                    "(--mode oracle/full)")
    pg.add_argument("--leaf-prob", type=float, default=0.3)
    pg.add_argument("--out-alias-prob", type=float, default=0.1,
                    help="chance a grow node's out= buffer aliases a live producer "
                         "(exercises compiler memory-planning) vs. a fresh allocation")
    pg.add_argument("--seed", type=int, default=0xC0FFEE)
    pg.add_argument("--slot", default="p0",
                    help="operational tag for stats/crash-marker filenames — not part of "
                         "any graph's identity, a graph is a pure function of (seed, index)")
    pg.add_argument("--start", type=int, default=0, help="first index to attempt")
    pg.add_argument("--backend", default="portable",
                    help="lowering target: portable (CPU kernels) | xnnpack | vulkan | ... "
                         "(only backends installed in this env; portable always works) "
                         "— --mode export/full")
    pg.add_argument("--quantize", action="store_true",
                    help="PT2E-quantize before lowering (backends that support it)")

    fd = sub.add_parser("feed", help="stream jobs to the broker, diff results, tally (a dir OR one graph)")
    fd.add_argument("jobs", nargs="*", help="specific job_id(s) or .job path(s) to replay (→ rich JSON); "
                                            "omit to feed the whole --corpus dir (→ TSV)")
    fd.add_argument("--host", default="127.0.0.1", help="broker host")
    fd.add_argument("--job-port", type=int, default=15554)
    fd.add_argument("--ctrl-port", type=int, default=15556)
    fd.add_argument("--corpus", default="tmp/corpus", help="corpus dir (also resolves bare job_ids)")
    fd.add_argument("--from-tsv", default=None, help="also take job_ids from a results TSV (col 2)")
    fd.add_argument("--status", default=None, help="with --from-tsv: keep only rows of this status")
    fd.add_argument("-n", "--graphs", type=int, default=0,
                    help="max jobs to feed; 0 = the whole corpus (also caps --from-tsv)")
    fd.add_argument("--timeout", type=float, default=30.0,
                    help="per-job timeout (a worker that never answers → TIMEOUT)")
    fd.add_argument("--window", type=int, default=64,
                    help="max in-flight jobs (end-to-end backpressure)")
    fd.add_argument("--skip-log", default="tmp/skip_reasons_mobile.tsv",
                    help="failing rows: status + job_id + reason + op-chain (graph is in the corpus)")
    fd.add_argument("--heartbeat", type=float, default=3.0,
                    help="seconds between feed status lines")
    fd.add_argument("-v", "--verbose", action="store_true")

    opts = ap.parse_args()

    if opts.cmd == "broker":
        from mobile.executor.broker import run_broker
        return run_broker(opts.job_port, opts.client_port, opts.ctrl_port,
                          opts.heartbeat, opts.verbose)
    if opts.cmd == "pregen":
        if opts.mode == "export" and not opts.oracle:
            ap.error("--mode export requires --oracle <dir>")
        from mobile.executor.pregen import run
        return run(opts.mode, opts.out, opts.oracle, opts.seed, opts.nodes, opts.leaf_prob,
                  opts.out_alias_prob, opts.backend, opts.quantize, opts.slot,
                  opts.start, opts.count, use_stdin=False)
    if opts.cmd == "feed":
        from mobile.executor.feed import run_feeder
        return run_feeder(opts.host, opts.job_port, opts.ctrl_port, opts.corpus,
                          opts.jobs, opts.from_tsv, opts.status, opts.graphs,
                          opts.timeout, opts.skip_log, opts.window,
                          opts.heartbeat, opts.verbose)
    raise SystemExit(f"unknown command: {opts.cmd}")  # argparse(required=True) makes this unreachable


if __name__ == "__main__":
    sys.exit(main())
