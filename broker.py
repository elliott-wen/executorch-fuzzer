#!/usr/bin/env python3
"""broker.py — start the ZeroMQ load-balancing broker.

Run it straight from the repo root, no -m / PYTHONPATH needed:

    python broker.py [-v] [--job-port 15554] [--client-port 15555] [--ctrl-port 15556]

The broker is a pure router: feeders connect on the job port, workers/clients (and phones)
on the client port. Start it first, then feed a corpus and attach any number of clients.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# Put the repo's PARENT on sys.path so `import mobile` resolves wherever this is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mobile.net.broker import run_broker


def main() -> int:
    ap = argparse.ArgumentParser(description="ZeroMQ load-balancing broker (feeders ↔ workers)")
    ap.add_argument("--host", help="(informational; the broker binds all interfaces)")
    ap.add_argument("--job-port", type=int, default=15554, help="frontend: feeders connect here")
    ap.add_argument("--client-port", type=int, default=15555, help="backend: workers/phones connect here")
    ap.add_argument("--ctrl-port", type=int, default=15556)
    ap.add_argument("--heartbeat", type=float, default=3.0,
                    help="seconds between broker status lines (with -v)")
    ap.add_argument("-v", "--verbose", action="store_true")
    opts = ap.parse_args()
    return run_broker(opts.job_port, opts.client_port, opts.ctrl_port,
                      opts.heartbeat, opts.verbose)


if __name__ == "__main__":
    sys.exit(main())
