"""feed.py — stream jobs to the broker AND own the diff + tally. The one entry point for both
bulk fuzzing and single-graph debugging.

The feeder reads one corpus — the lowering stage puts the .pte and the oracle record side by
side — so it already holds the reference, sends each LEAN job
(pte + inputs) to the broker, and the broker routes the worker's RAW-output RESULT back HERE.
The feeder then diffs outputs vs its kept eager ([compare.py]), tallies, and writes the skip-log.
Comparison lives with the data that needs it (the feeder); the broker stays a pure router; the
worker (a local `client`, or the phone) just runs + returns. Run more feeders to parallelise.

Two ways to point it at work — same pipeline, same compare:
    python -m executor.feed --corpus tmp/pte          # everything that lowered → TSV + tally
    python -m executor.feed <token> --corpus tmp/pte  # one graph → rich JSON observation
    python -m executor.feed --from-tsv tmp/skip.tsv --status MISMATCH -n 20  # replay failures

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

from mobile.generator.lower import store as pte_store
from mobile.generator.oracle import store as oracle_store
from mobile.executor import compare as cmp
from mobile.executor import protocol as P

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


def _worklist(pte_dir, jobs, from_tsv, status_filter, limit):
    """Return (tokens, debug). Explicit tokens / --from-tsv → debug (rich JSON); otherwise
    every graph that lowered → bulk (TSV).

    The bulk list comes from the .pte corpus rather than the oracle one: a graph only has
    something to run if lowering produced a program for it, so this is exactly the work that
    exists. Nothing needs to ask the oracle corpus what it holds."""
    tokens = list(jobs or [])
    if from_tsv:
        tokens += _ids_from_tsv(from_tsv, status_filter, limit)
    if tokens:                                            # debug: specific graphs
        return tokens, True
    tokens = list(pte_store.iter_tokens(pte_dir))          # bulk: everything that lowered
    if limit > 0:
        tokens = tokens[:limit]
    return tokens, False


def _load_job(pte_dir, token):
    """One token → the pushjob frames to ship, plus what the feeder keeps to diff against.

    Everything comes from the one corpus: the lowering stage copies the oracle record in
    beside the .pte, and the two stores share a layout, so the oracle reader works on this
    directory unchanged. The executor is handed one path and a corpus travels whole.
    """
    lowered = pte_store.read_record(pte_dir, token)
    if lowered is None:
        raise FileNotFoundError(f"no .pte for {token} in {pte_dir}")
    record = oracle_store.read_record(pte_dir, token)
    if record is None:
        raise FileNotFoundError(f"no oracle record beside the .pte for {token} in {pte_dir}")
    delegation = lowered.get("delegation")
    return P.encode_pushjob(
        token, lowered["pte"], record["inputs"], record["eager"],
        desc=record.get("desc", ""), user_pos=lowered.get("user_outputs"),
        delegated=({"ops": delegation["delegated"], "non": delegation["portable"],
                    "calls": delegation["calls"]} if delegation else None),
    )


# ── main loop ────────────────────────────────────────────────────────────────────

def run_feeder(host, job_port, ctrl_port, pte_dir, jobs=None, from_tsv=None,
               status_filter=None, limit=0, timeout=30.0,
               skip_log="tmp/skip_reasons_mobile.tsv", window=64,
               heartbeat=3.0, verbose=False) -> int:
    work, debug = _worklist(pte_dir, jobs, from_tsv, status_filter, limit)
    if not work:
        print(f"[feed] nothing to do — give a token, --from-tsv, or a populated --corpus {pte_dir}",
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
                token = work[wi]
                wi += 1
                try:
                    frames = _load_job(pte_dir, token)
                    job_id, desc = P.peek_jobinfo(frames)
                    eager, user_pos = P.eager_from_pushjob(frames)
                except FileNotFoundError as e:
                    record("NOJOB", str(e), token, "")
                    continue
                except Exception as e:
                    record("SKIP", f"feed read {type(e).__name__}: {e}", token, "")
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
        print(f"  skip-log: {skip_log}  (replay one with `python -m executor.feed <job_id>`)")
    return 1 if (tally["MISMATCH"] or tally["CRASH"]) else 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m executor.feed",
                                 description="stream jobs to the broker, diff results, tally "
                                             "(a whole corpus OR one graph)")
    ap.add_argument("jobs", nargs="*", help="specific job_id(s) or .job path(s) to replay "
                                            "(→ rich JSON); omit to feed everything that lowered "
                                            "(→ TSV)")
    ap.add_argument("--host", default="127.0.0.1", help="broker host")
    ap.add_argument("--job-port", type=int, default=15554)
    ap.add_argument("--ctrl-port", type=int, default=15556)
    ap.add_argument("--corpus", default="tmp/pte",
                    help="lowering corpus: .pte plus the oracle record copied beside it")
    ap.add_argument("--from-tsv", default=None, help="also take job_ids from a results TSV (col 2)")
    ap.add_argument("--status", default=None, help="with --from-tsv: keep only rows of this status")
    ap.add_argument("-n", "--graphs", type=int, default=0,
                    help="max jobs to feed; 0 = everything (also caps --from-tsv)")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="per-job timeout (a worker that never answers → TIMEOUT)")
    ap.add_argument("--window", type=int, default=64,
                    help="max in-flight jobs (end-to-end backpressure)")
    ap.add_argument("--skip-log", default="tmp/skip_reasons_mobile.tsv",
                    help="failing rows: status + token + reason + op-chain "
                         "(the graph is in the corpus)")
    ap.add_argument("--heartbeat", type=float, default=3.0,
                    help="seconds between feed status lines")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    return run_feeder(a.host, a.job_port, a.ctrl_port, a.corpus, a.jobs, a.from_tsv,
                      a.status, a.graphs, a.timeout, a.skip_log, a.window,
                      a.heartbeat, a.verbose)


if __name__ == "__main__":
    import sys
    sys.exit(main())
