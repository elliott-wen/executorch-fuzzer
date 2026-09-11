"""broker.py — a pure load-balancing ROUTER↔ROUTER broker. It shuffles envelopes; it NEVER
touches a tensor, compares, or tallies.

Two bound ROUTER sockets + a PUB:
  frontend (jobs port)    ↔ FEEDERS (DEALER)  — send lean jobs, receive results back
  backend  (clients port) ↔ WORKERS (REQ)     — LRU work-pull (READY → job → RESULT)
  ctrl     (ctrl port)    → STOP broadcast on shutdown

The feeder holds the eager reference (it read the corpus) and does the diff itself, so the broker
only needs to get each worker's RESULT back to the feeder that sent the job. It does that by peeking
the cleartext job_id in the (uncompressed) JSON header — the same id it reads off the job — and never
decompresses a payload frame. State is minimal and routing-only: an LRU queue of free workers and a
job_id→feeder map. Scale feeders (each diffs its own jobs) and workers; the broker is just the switch.
"""

from __future__ import annotations

import json
import time
from collections import deque

import zmq


def _job_id(frames: list[bytes]) -> str:
    """Read job_id from a job's or result's cleartext JSON header (frame 0). No payload touched."""
    return json.loads(frames[0].decode("utf-8"))["job_id"]


def run_broker(job_port: int, client_port: int, ctrl_port: int,
               heartbeat: float = 3.0, verbose: bool = False, **_ignored) -> int:
    ctx = zmq.Context.instance()
    # ZMTP heartbeats reap a silently-dropped peer (a phone that lost network, a feeder that died)
    # instead of leaving a half-open connection forever. ROUTER_MANDATORY makes a send to a
    # reaped/unknown peer raise EHOSTUNREACH so we can drop the ghost instead of losing the message.
    frontend = ctx.socket(zmq.ROUTER)                 # feeders (DEALER)
    backend = ctx.socket(zmq.ROUTER)                  # workers (REQ)
    for s in (frontend, backend):
        s.setsockopt(zmq.HEARTBEAT_IVL, 5000)
        s.setsockopt(zmq.HEARTBEAT_TIMEOUT, 20000)
        s.setsockopt(zmq.HEARTBEAT_TTL, 20000)
        s.setsockopt(zmq.ROUTER_MANDATORY, 1)
    frontend.bind(f"tcp://*:{job_port}")
    backend.bind(f"tcp://*:{client_port}")
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{ctrl_port}")

    free_workers: deque = deque()        # worker identities awaiting work (LRU)
    inflight: dict = {}                  # job_id -> (feeder_identity, dispatch_time)

    # Poll the frontend ONLY when a worker is free — otherwise unread feeder jobs stay in the
    # ROUTER buffer (end-to-end backpressure to the feeder) instead of piling up in the broker.
    p_both = zmq.Poller(); p_both.register(backend, zmq.POLLIN); p_both.register(frontend, zmq.POLLIN)
    p_back = zmq.Poller(); p_back.register(backend, zmq.POLLIN)

    print(f"Broker up — feeders tcp://*:{job_port}  workers tcp://*:{client_port}  "
          f"ctrl tcp://*:{ctrl_port}", flush=True)
    print(f"  feed  :  python -m mobile feed   --host <broker-ip> --job-port {job_port} --ctrl-port {ctrl_port} --corpus tmp/corpus", flush=True)
    print(f"  client:  python local_client/xnnpack_client/xnnpack_client.py --host <broker-ip> --client-port {client_port} --ctrl-port {ctrl_port}", flush=True)

    t0 = time.monotonic()
    last_beat = t0
    routed = 0

    def dispatch_to_worker(job_frames: list[bytes], feeder: bytes) -> bool:
        """Send a job to the next free worker; returns False if none accepted it."""
        while free_workers:
            worker = free_workers.popleft()
            try:
                backend.send_multipart([worker, b"", *job_frames])
                inflight[_job_id(job_frames)] = (feeder, time.monotonic())
                return True
            except zmq.ZMQError as e:
                if e.errno == zmq.EHOSTUNREACH:
                    continue                          # ghost worker (heartbeat-reaped) — try the next
                raise
        return False

    try:
        while True:
            poller = p_both if free_workers else p_back
            socks = dict(poller.poll(timeout=500))

            if backend in socks:
                msg = backend.recv_multipart()        # [worker, b'', tag, ...]
                worker, content = msg[0], msg[2:]
                tag = content[0] if content else b""
                if tag == b"READY":
                    free_workers.append(worker)
                elif tag == b"RESULT":
                    result_frames = content[1:]       # [header, out_raws...] — forwarded as-is
                    try:
                        feeder, _ = inflight.pop(_job_id(result_frames), (None, None))
                    except Exception:
                        feeder = None
                    if feeder is not None:
                        try:
                            frontend.send_multipart([feeder, *result_frames])
                            routed += 1
                            if verbose:
                                print(f"    → routed result {_job_id(result_frames)} to feeder", flush=True)
                        except zmq.ZMQError:
                            pass                      # feeder gone — drop the late result
                    free_workers.append(worker)       # worker is free again

            if frontend in socks and free_workers:
                msg = frontend.recv_multipart()       # [feeder, *job_frames]  (DEALER: no empty)
                feeder, job_frames = msg[0], msg[1:]
                if not dispatch_to_worker(job_frames, feeder):
                    pass                              # no worker accepted — feeder will time out & resend

            now = time.monotonic()
            # Expire routing entries for jobs whose worker died (the feeder times these out itself);
            # this just keeps the map from leaking. Generous bound — far past any real job.
            if now - last_beat >= heartbeat:
                for jid, (_f, ts) in list(inflight.items()):
                    if now - ts > 600:
                        inflight.pop(jid, None)
                last_beat = now
                el = now - t0
                # Liveness line every heartbeat. free=0 → no workers connected; inflight stuck → a
                # worker is mid-run or died (the feeder will time it out). -v adds the per-message detail.
                print(f"  [t={el:4.0f}s] routed={routed}  workers={len(free_workers)} free  "
                      f"inflight={len(inflight)}", flush=True)
    except KeyboardInterrupt:
        print("\n  ^C — shutting down the fleet ...", flush=True)
    finally:
        pub.send(b"STOP")
        time.sleep(0.3)                               # let STOP propagate to the fleet

    print(f"\n── Broker done — routed {routed} results ──", flush=True)
    return 0
