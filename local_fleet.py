#!/usr/bin/env python3
"""local_fleet.py — LOCAL test fleet: spawn + auto-respawn the executor fleet (feeder + clients).

One orchestrator for the run-time side of the pre-gen flow: it launches an optional
corpus feeder plus M clients as subprocesses, restarts any that exit (with backoff so a
broken worker can't hot-loop), prints a periodic status line, and shuts everything down
cleanly on Ctrl-C.

  # offline: build the corpus first
  python mobile/pregen_fleet.py --workers 64 --per-worker 200 --out tmp/corpus
  # then, the live fleet:
  python -m mobile broker                                          # terminal 1
  python mobile/local_fleet.py --clients 16 --corpus tmp/corpus      # terminal 2 (feeder + clients)

Pass --corpus to also run a one-shot feeder (feeds the corpus once, diffs + tallies, then exits —
not respawned); omit it to feed yourself. The broker is NOT managed here — start it first; it's a
pure router that runs until Ctrl-C (the feeder stops itself when the corpus is drained).
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / "venv" / "bin" / "python")
ENV = {**os.environ, "PYTHONPATH": str(REPO), "CUDA_VISIBLE_DEVICES": ""}


class Worker:
    """One supervised subprocess (the feeder or a client) + its restart policy."""

    def __init__(self, name: str, cmd: list[str], log_path: Path,
                 min_alive: float, backoff: float, max_backoff: float,
                 restart: bool = True):
        self.name = name
        self.cmd = cmd
        self.log_path = log_path
        self.min_alive = min_alive          # lived < this ⇒ count as a fast death
        self.backoff = backoff              # base backoff per consecutive fast death
        self.max_backoff = max_backoff
        self.restart = restart              # False ⇒ one-shot (the feeder): never respawn
        self.proc: subprocess.Popen | None = None
        self.started = 0.0
        self.restarts = 0
        self.fast_deaths = 0
        self.respawn_at = 0.0               # monotonic time to (re)start; 0 = now
        self.done = False                   # one-shot finished ⇒ leave it stopped

    def start(self) -> None:
        fh = open(self.log_path, "a")
        fh.write(f"\n===== start {self.name} @ {time.strftime('%H:%M:%S')} "
                 f"(restart #{self.restarts}) =====\n")
        fh.flush()
        self._fh = fh
        self.proc = subprocess.Popen(self.cmd, stdout=fh, stderr=subprocess.STDOUT,
                                     env=ENV, cwd=str(REPO))
        self.started = time.monotonic()
        self.respawn_at = 0.0

    def note_exit(self, rc: int, now: float) -> float:
        """Record an exit and schedule the respawn. Returns the backoff applied."""
        try:
            self._fh.close()
        except Exception:
            pass
        self.proc = None
        if not self.restart:                # one-shot (feeder): finished, never respawn
            self.done = True
            return 0.0
        alive = now - self.started
        self.restarts += 1
        if alive < self.min_alive:
            self.fast_deaths += 1
        else:
            self.fast_deaths = 0            # a healthy run resets the backoff
        bo = min(self.max_backoff, self.fast_deaths * self.backoff)
        self.respawn_at = now + bo
        return bo

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


def build_workers(a) -> list[Worker]:
    workers: list[Worker] = []
    logdir = Path(a.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    common = dict(min_alive=a.min_alive, backoff=a.backoff, max_backoff=a.max_backoff)
    if a.corpus:
        cmd = [PY, "-m", "mobile", "feed", "--host", a.host,
               "--job-port", str(a.job_port), "--ctrl-port", str(a.ctrl_port),
               "--corpus", a.corpus]
        # one-shot: feeds the corpus once, then exits — never respawned.
        workers.append(Worker("feeder", cmd, logdir / "feeder.log", restart=False, **common))
    for i in range(a.clients):
        cmd = [PY, "-m", "mobile", "client", "--host", a.host,
               "--client-port", str(a.client_port), "--ctrl-port", str(a.ctrl_port),
               "--label", f"c{i}", "--job-timeout", str(a.job_timeout)]
        workers.append(Worker(f"client-{i}", cmd, logdir / f"client-{i}.log", **common))
    return workers


def main() -> int:
    ap = argparse.ArgumentParser(description="supervise the mobile feeder + client fleet")
    ap.add_argument("--clients", type=int, default=128)
    ap.add_argument("--corpus", default="", help="if set, run a one-shot feeder of this corpus (feeds once, exits)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--job-port", type=int, default=15554)
    ap.add_argument("--client-port", type=int, default=15555)
    ap.add_argument("--ctrl-port", type=int, default=15556)
    ap.add_argument("--job-timeout", type=float, default=30.0)
    ap.add_argument("--stagger", type=float, default=0.2, help="seconds between launches")
    ap.add_argument("--poll", type=float, default=2.0, help="seconds between liveness checks")
    ap.add_argument("--status", type=float, default=15.0, help="seconds between status lines")
    ap.add_argument("--min-alive", type=float, default=15.0,
                    help="a worker dying younger than this counts toward backoff")
    ap.add_argument("--backoff", type=float, default=3.0, help="backoff per consecutive fast death")
    ap.add_argument("--max-backoff", type=float, default=60.0)
    ap.add_argument("--logdir", default="tmp/fleet_logs")
    a = ap.parse_args()

    workers = build_workers(a)
    stop = {"flag": False}

    def on_sig(_signum, _frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    roles = f"{a.clients} clients" + (" + feeder" if a.corpus else "")
    print(f"supervising {roles} → broker {a.host}  (logs: {a.logdir}/)", flush=True)

    # staggered initial launch
    for w in workers:
        if stop["flag"]:
            break
        w.start()
        time.sleep(a.stagger)

    last_status = 0.0
    while not stop["flag"]:
        now = time.monotonic()
        for w in workers:
            if stop["flag"]:
                break
            if w.proc is not None:
                rc = w.proc.poll()
                if rc is not None:                      # just exited
                    bo = w.note_exit(rc, now)
                    print(f"[{time.strftime('%H:%M:%S')}] {w.name} exited rc={rc} "
                          f"(restart #{w.restarts}"
                          f"{f', backoff {bo:.0f}s' if bo else ''})", flush=True)
            elif not w.done and now >= w.respawn_at:    # dead and due for respawn
                w.start()

        if now - last_status >= a.status:
            last_status = now
            running = sum(1 for w in workers if w.proc and w.proc.poll() is None)
            restarts = sum(w.restarts for w in workers)
            print(f"[{time.strftime('%H:%M:%S')}] up {running}/{len(workers)}  "
                  f"total_restarts={restarts}", flush=True)

        time.sleep(a.poll)

    # graceful shutdown
    print("\nstopping fleet ...", flush=True)
    for w in workers:
        w.stop()
    deadline = time.monotonic() + 10
    for w in workers:
        if w.proc:
            try:
                w.proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                w.proc.kill()
    print("done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
