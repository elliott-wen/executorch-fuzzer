#!/usr/bin/env python3
"""nf_full.py <backend> <outfile> [--mech M] [--shard i --nshards n]
Population sweep: for EVERY row of a mechanism bucket, reconstruct BOTH the A-side and the
B-side graph and run them on <backend>. For portable (deleg_ops==0 by construction) this
answers "does the divergence exist with no delegate at all?" over the whole bucket.
Runs each graph in a forked child so a native abort costs one graph, not the sweep.
"""
import os, sys, json, argparse, multiprocessing as mp
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from nf_common import load_rows, srcs_for, leaf_tensors, nf_signature, classify
from mobile.gen.export import build_job
from mobile.executor import compare as cmp


def _work(q, src, pos, backend):
    try:
        j = build_job(src, backend, False)
        if j.status != "READY":
            q.put(dict(res=f"BUILD:{j.status}")); return
        d = dict(d=j.delegated_ops, nd=j.non_delegated_ops, calls=j.delegate_calls)
        from mobile.executor.et_runner import run_pte
        raw = run_pte(j.pte, j.inputs)
        et = cmp.select(list(raw), j.user_pos)
        if len(et) > len(j.eager):
            et = et[len(et) - len(j.eager):]
        p = pos if pos >= 0 else len(j.eager) - 1
        ref, dev = j.eager[p], et[p]
        if not isinstance(dev, torch.Tensor):
            d["res"] = "NONTENSOR"; q.put(d); return
        st, mech, kind = classify(ref, dev)
        d.update(res=st, mech=mech, kind=kind, dtype=str(ref.dtype).replace("torch.", ""),
                 ref_nf=nf_signature(ref), dev_nf=nf_signature(dev))
        q.put(d)
    except BaseException as e:
        q.put(dict(res=f"EXC:{type(e).__name__}:{str(e)[:80]}"))


def run_one(src, pos, backend, timeout=180):
    q = mp.Queue()
    p = mp.Process(target=_work, args=(q, src, pos, backend))
    p.start(); p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(); return dict(res="TIMEOUT")
    if not q.empty():
        return q.get()
    return dict(res=f"CRASH:exit={p.exitcode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend"); ap.add_argument("out")
    ap.add_argument("--mech", default="NONFINITE")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    rows = [r for r in load_rows() if r["mechanism"] == a.mech]
    rows.sort(key=lambda r: (r["job"], r["out"]))
    mine = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                d = json.loads(l); done.add((d["job"], d["out"]))
            except Exception:
                pass
    fh = open(a.out, "a", buffering=1)
    for r in mine:
        if (r["job"], r["out"]) in done:
            continue
        rec = dict(job=r["job"], out=r["out"], verdict=r["verdict"], tgt_op=r["tgt_op"],
                   backend=a.backend)
        try:
            srcA, srcB, target, posA, posB = srcs_for(r["job"], r["out"], r["verdict"], r["needed"])
            lt = leaf_tensors(r["job"])
            bad = [n for n, t in lt.items()
                   if t.dtype.is_floating_point and not bool(torch.isfinite(t.float()).all())]
            rec["leaf_nonfinite"] = ",".join(sorted(bad)) or "no"
            rec["A"] = run_one(srcA, posA, a.backend)
            rec["B"] = run_one(srcB, posB, a.backend)
        except BaseException as e:
            rec["ERR"] = f"{type(e).__name__}:{str(e)[:100]}"
        fh.write(json.dumps(rec) + "\n")
        print(f"[{a.backend} s{a.shard}] {r['job']}:{r['out']} {r['tgt_op']} "
              f"A={rec.get('A',{}).get('res')} B={rec.get('B',{}).get('res')} "
              f"Bd={rec.get('B',{}).get('d')} Bkind={rec.get('B',{}).get('kind')} {rec.get('ERR','')}",
              flush=True)
    print(f"DONE {a.backend} s{a.shard}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
