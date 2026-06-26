"""feed.py — stream jobs to the broker AND own the diff + tally. The one entry point for both
bulk fuzzing and single-graph debugging.

The feeder reads the corpus (so it already holds the eager reference), sends each LEAN job
(pte + inputs) to the broker, and the broker routes the worker's RAW-output RESULT back HERE.
The feeder then diffs outputs vs its kept eager ([compare.py]), tallies, and writes the skip-log.
Comparison lives with the data that needs it (the feeder); the broker stays a pure router; the
worker (a local `client`, or the phone) just runs + returns. Run more feeders to parallelise.

Two ways to point it at work — same pipeline, same compare:
    mobile feed --corpus tmp/corpus              # the whole corpus → TSV + tally (bulk fuzzing)
    mobile feed w33:531 --corpus tmp/corpus      # one graph (job_id) → rich JSON observation
    mobile feed path/to/w33_531.job              # one graph (direct .job path)
    mobile feed --from-tsv tmp/skip.tsv --status MISMATCH --limit 20   # replay failing rows

Single-graph / --from-tsv runs print one JSON object per job (job_id, status, detail, op_chain,
per-output {dtype, shape, max_abs_delta, eager, et}) — the debug observation. Where it RAN is just
which worker is connected: a local host executor (xnnpack_client, x86) or the phone (device, ARM).

Transport: DEALER ↔ broker frontend ROUTER. A bounded in-flight WINDOW gives end-to-end
backpressure. A job with no result within the timeout is recorded TIMEOUT (a dead/hung worker).
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import time
from pathlib import Path

import zmq

from mobile.net import corpus as C
from mobile.net import compare as cmp
from mobile.net import protocol as P

_TALLY_KEYS = ("OK", "MISMATCH", "CRASH", "TIMEOUT", "SKIP")


# ── work-list resolution ─────────────────────────────────────────────────────────

def _ids_from_tsv(path: str, status_filter: str | None, limit: int) -> list[str]:
    ids: list[str] = []
    with open(path) as f:
        next(f, None)                                     # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            st, jid = parts[0], parts[1]
            if status_filter and st != status_filter:
                continue
            ids.append(jid)
            if limit and len(ids) >= limit:
                break
    return ids


def _worklist(corpus_dir, jobs, from_tsv, status_filter, limit):
    """Return ([(job_id_hint, path)], debug). Explicit jobs / --from-tsv → debug (rich JSON);
    otherwise the whole corpus dir → bulk (TSV)."""
    ids = list(jobs or [])
    if from_tsv:
        ids += _ids_from_tsv(from_tsv, status_filter, limit)
    if ids:                                               # debug: specific graphs
        out = []
        for j in ids:
            p = Path(j)
            path = p if (p.suffix == ".job" and p.exists()) else C.job_file(corpus_dir, j)
            out.append((j, Path(path)))
        return out, True
    files = C.iter_job_files(corpus_dir)                  # bulk: the whole corpus
    if limit > 0:
        files = files[:limit]
    return [(None, fp) for fp in files], False


# ── main loop ────────────────────────────────────────────────────────────────────

def run_feeder(host, job_port, ctrl_port, corpus_dir, jobs=None, from_tsv=None,
               status_filter=None, limit=0, timeout=30.0,
               skip_log="tmp/skip_reasons_mobile.tsv", window=64,
               heartbeat=3.0, verbose=False) -> int:
    work, debug = _worklist(corpus_dir, jobs, from_tsv, status_filter, limit)
    if not work:
        print(f"[feed] nothing to do — give a job_id, --from-tsv, or a populated --corpus {corpus_dir}",
              flush=True)
        return 1
    if debug:
        window = min(window, max(1, len(work)))           # don't over-buffer a tiny debug set

    ctx = zmq.Context.instance()
    dealer = ctx.socket(zmq.DEALER)
    dealer.set_hwm(max(256, window * 8))
    dealer.setsockopt(zmq.HEARTBEAT_IVL, 5000)
    dealer.setsockopt(zmq.HEARTBEAT_TIMEOUT, 20000)
    dealer.setsockopt(zmq.HEARTBEAT_TTL, 20000)
    dealer.connect(f"tcp://{host}:{job_port}")
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{host}:{ctrl_port}")
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    poller = zmq.Poller()
    poller.register(dealer, zmq.POLLIN)
    poller.register(sub, zmq.POLLIN)

    tally = {k: 0 for k in _TALLY_KEYS}
    pending: dict = {}                                    # job_id -> (eager, user_pos, desc, sent_time)
    state = {"executed": 0}

    skip_fh = None
    if not debug:
        lp = Path(skip_log)
        lp.parent.mkdir(parents=True, exist_ok=True)
        skip_fh = open(lp, "w")
        skip_fh.write("status\tjob\treason\tgraph\n")
        skip_fh.flush()

    def record(status, detail, job_id, desc, eager=None, et=None):
        tally[status] = tally.get(status, 0) + 1
        state["executed"] += 1
        if debug:                                         # single-graph: print the rich observation
            rec = {"job_id": job_id, "status": status, "detail": detail, "op_chain": desc}
            if status in ("OK", "MISMATCH") and eager is not None and et is not None:
                rec["outputs"] = cmp.report(eager, et)
            print(json.dumps(rec), flush=True)
        else:                                             # bulk: TSV + periodic tally
            if status != "OK":
                def _r(s):
                    return (s or "").replace("\t", " ").replace("\n", " ").replace("\r", " ")
                skip_fh.write(f"{status}\t{job_id}\t{_r(detail)}\t{_r(desc)}\n")
                skip_fh.flush()
            if verbose or status in ("MISMATCH", "CRASH"):
                print(f"  [{state['executed']}] {status:8s} {job_id} {(detail or '')[:80]}", flush=True)

    print(f"[feed] {len(work)} {'graph(s) [debug]' if debug else 'jobs'} "
          f"→ broker {host}:{job_port} (window={window}, timeout={timeout:.0f}s)", flush=True)

    t0 = time.monotonic()
    last_beat = t0
    tgt = len(work)

    def beat():
        if debug:
            return
        el = time.monotonic() - t0
        n = state["executed"]
        rate = n / el if el > 0 else 0.0
        t = tally
        print(f"  [t={el:4.0f}s] {n}/{tgt} ({rate:4.1f}/s)  ok={t['OK']} mism={t['MISMATCH']} "
              f"crash={t['CRASH']} timeout={t['TIMEOUT']} skip={t['SKIP']}  |  inflight={len(pending)}",
              flush=True)

    wi = 0
    try:
        while pending or wi < len(work):
            while len(pending) < window and wi < len(work):
                hint, path = work[wi]
                wi += 1
                if not path.exists():
                    record("NOJOB", f"not found in corpus {corpus_dir}", hint or str(path), "")
                    continue
                try:
                    frames = C.read_job(path)
                    job_id, desc = P.peek_jobinfo(frames)
                    eager, user_pos = P.eager_from_pushjob(frames)
                except Exception as e:
                    record("SKIP", f"feed read {type(e).__name__}: {e}", hint or path.name, "")
                    continue
                pending[job_id] = (eager, user_pos, desc, time.monotonic())
                dealer.send_multipart(P.job_frames_from_pushjob(frames))

            socks = dict(poller.poll(timeout=500))
            if sub in socks and sub.recv() == b"STOP":
                print("[feed] STOP received from broker", flush=True)
                break

            if dealer in socks:
                result_frames = dealer.recv_multipart()
                try:
                    job_id, status, detail, outputs = P.decode_result(result_frames)
                except Exception:
                    continue
                info = pending.pop(job_id, None)
                if info is not None:
                    eager, user_pos, desc, _ = info
                    if status == "RAN":                       # worker ran clean → diff HERE
                        et = cmp.select(outputs, user_pos)
                        try:
                            status, detail = cmp.compare(eager, et)
                        except Exception as e:
                            status, detail = "SKIP", f"feed compare {type(e).__name__}: {e}"
                        record(status, detail, job_id, desc, eager, et)
                    else:                                     # SKIP / CRASH / TIMEOUT from the worker
                        record(status, detail, job_id, desc)

            now = time.monotonic()
            for job_id, (_eg, _up, desc, sent) in list(pending.items()):
                if now - sent > timeout * 2:
                    pending.pop(job_id, None)
                    record("TIMEOUT", f"no result within {timeout * 2:.0f}s", job_id, desc)
            if now - last_beat >= heartbeat:
                last_beat = now
                beat()
    except KeyboardInterrupt:
        print("\n[feed] ^C", flush=True)
    finally:
        if skip_fh is not None:
            skip_fh.close()

    if not debug:
        print(f"\n── Done — {state['executed']} graphs {'─' * 30}")
        for k in _TALLY_KEYS:
            print(f"  {k:9s}: {tally[k]}")
        print(f"  skip-log: {skip_log}  (replay one with `mobile feed <job_id>`)")
    return 1 if (tally["MISMATCH"] or tally["CRASH"]) else 0
