#!/usr/bin/env python3
"""bisect_driver.py — full two-axis localization of every CoreML MISMATCH (analysis.md step 2).

Per job (job_id, out_idx): bisect_output_set → CONE_LOCAL / SIBLING_DEPENDENT; if CONE_LOCAL,
bisect_cone → OPERATOR / COMPOSITIONAL. One resumable results row per (job, out). Sharded: run N
copies with --shard i --nshards N, each its own results file + DEALER, all → the one broker.

Lowering (build_job "coreml") runs HERE on Linux; the .pte executes on the connected Mac worker
via the broker. Env: BROKER_PORT (feeder port, default 15554), ISO_DEV_TIMEOUT.

  python bisect_driver.py --shard 0 --nshards 6 --out bisect_results
"""
import os, sys, time, warnings, logging, argparse, re, zlib
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
sys.path.insert(0, "/data/jwen929")
import zmq
from mobile.gen.export import build_job
from mobile.executor import protocol as P, compare as cmp, corpus as C
from mobile.gen.diff.bisect import bisect_output_set, bisect_cone, CONE_LOCAL
from mobile.gen.diff.cone_predicate import cone_localizer, output_set_localizer

BK = "coreml"
CORPUS = "/data/jwen929/mobile/corpus_v4/coreml"
PORT = int(os.environ.get("BROKER_PORT", "15554"))
TMO = float(os.environ.get("ISO_DEV_TIMEOUT", "120"))

ctx = zmq.Context.instance(); D = ctx.socket(zmq.DEALER); D.setsockopt(zmq.HEARTBEAT_IVL, 5000)
D.connect(f"tcp://127.0.0.1:{PORT}"); POL = zmq.Poller(); POL.register(D, zmq.POLLIN); seq = [0]

def _run(src):
    """Lower src to coreml, run on the Mac via broker, return (job, raw_outputs). Raises on fail."""
    job = build_job(src, BK)
    if job.status != "READY":
        raise RuntimeError(f"BUILD:{job.status}: {job.detail[:120]}")   # keep the reason, not just SKIP
    seq[0] += 1
    fr = P.encode_pushjob(f"bs::{seq[0]}", job.pte, job.inputs, job.eager, desc="bs", user_pos=job.user_pos)
    D.send_multipart(P.job_frames_from_pushjob(fr)); t0 = time.monotonic()
    while time.monotonic() - t0 < TMO:
        if dict(POL.poll(1000)).get(D):
            _, st, det, raw = P.decode_result(D.recv_multipart())
            if st != "RAN":
                raise RuntimeError(f"DEV:{st}")
            return job, raw
    raise RuntimeError("DEV:TIMEOUT")

def _diverges_at(job, raw, tpos):
    et = cmp.select(raw, job.user_pos)
    if len(et) > len(job.eager):
        et = et[len(et) - len(job.eager):]
    return cmp._cmp(job.eager[tpos], et[tpos], 0)[0] == "MISMATCH"

def localize(job_id, out_idx):
    """Full localization → (verdict, detail_dict). verdict ∈ {SIBLING_DEPENDENT, OPERATOR,
    COMPOSITIONAL, INCONCLUSIVE, LOWERFAIL, ERROR}."""
    py = C.job_file(CORPUS, job_id, "py")
    # --- axis 1: output set ---
    others, build_output_src, target, ret = output_set_localizer(py, out_idx)
    def target_diverges(returned):
        sel = [o for o in ret if o in set(returned)]
        job, raw = _run(build_output_src(returned))
        return _diverges_at(job, raw, sel.index(target))
    os_v = bisect_output_set(target, others, target_diverges)
    if os_v.verdict == "SIBLING_DEPENDENT":
        return "SIBLING_DEPENDENT", {"needed_siblings": os_v.needed_siblings, "cone": len(others)}
    if os_v.verdict == "INCONCLUSIVE":
        return "INCONCLUSIVE", {"stage": "output_set"}
    # CONE_LOCAL → axis 2: cone
    anc, build_src, tgt = cone_localizer(py, out_idx)
    def diverges_with_live(kept):
        job, raw = _run(build_src(kept))
        et = cmp.select(raw, job.user_pos)
        if len(et) > len(job.eager): et = et[len(et) - len(job.eager):]
        return cmp._cmp(job.eager[-1], et[-1], 0)[0] == "MISMATCH"
    try:
        c_v = bisect_cone(tgt, anc, diverges_with_live)
    except RuntimeError as e:
        # target (or a reduced cone) won't lower ALONE on this strict-partitioning backend →
        # can't split operator vs compositional. It's still CONE_LOCAL (no sibling needed), just
        # not isolable — NOT a graph-opt-by-sibling finding. Preserve that, don't drop to LOWERFAIL.
        if str(e).startswith("BUILD:"):
            return "CONE_LOCAL_CANT_SPLIT", {"cone": len(anc), "stage": "cone_build_skip"}
        raise
    return c_v.verdict, {"needed_live": c_v.needed_live, "cone": len(anc)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--skiplog", default="/data/jwen929/mobile/findings/coreml_mac2/skiplog.tsv")
    ap.add_argument("--out", default="/data/jwen929/mobile/findings/coreml_mac2/bisect_results")
    a = ap.parse_args()
    # worklist: MISMATCH rows → (job_id, out_idx from "out[N]")
    work = []
    for ln in open(a.skiplog).read().splitlines()[1:]:
        p = ln.split("\t")
        if len(p) < 3 or p[0] != "MISMATCH": continue
        m = re.search(r"out\[(\d+)\]", p[2])
        work.append((p[1], int(m.group(1)) if m else 0))
    # STABLE shard assignment (crc32 of job:out) so resuming across a growing skiplog neither
    # re-bisects nor drops cases as the worklist order shifts.
    work = [w for w in work if zlib.crc32(f"{w[0]}:{w[1]}".encode()) % a.nshards == a.shard]
    outf = f"{a.out}.{a.shard}.tsv"
    done = set()
    if os.path.exists(outf):
        for ln in open(outf).read().splitlines()[1:]:
            c = ln.split("\t")
            if len(c) >= 3 and c[2] not in ("LOWERFAIL", "ERROR", "DEV:TIMEOUT"):
                done.add((c[0], int(c[1])))
    fh = open(outf, "a")
    if os.path.getsize(outf) == 0:
        fh.write("job_id\tout_idx\tverdict\tcone_size\trequired\tdetail\n"); fh.flush()
    for k, (jid, oi) in enumerate(work):
        if (jid, oi) in done: continue
        t0 = time.monotonic()
        try:
            verdict, d = localize(jid, oi)
            req = d.get("needed_siblings") or d.get("needed_live") or []
            fh.write(f"{jid}\t{oi}\t{verdict}\t{d.get('cone','')}\t{','.join(req)}\t{d.get('stage','')}\n")
        except Exception as e:
            msg = str(e).replace("\t", " ").replace("\n", " ")[:80]
            v = "LOWERFAIL" if msg.startswith("BUILD:") else ("DEV:TIMEOUT" if "TIMEOUT" in msg else "ERROR")
            fh.write(f"{jid}\t{oi}\t{v}\t\t\t{msg}\n")
        fh.flush()
        if k % 20 == 0:
            print(f"[shard {a.shard}] {k}/{len(work)} last={jid} ({time.monotonic()-t0:.1f}s)", flush=True)
    print(f"[shard {a.shard}] DONE {len(work)} jobs", flush=True)

if __name__ == "__main__":
    main()
