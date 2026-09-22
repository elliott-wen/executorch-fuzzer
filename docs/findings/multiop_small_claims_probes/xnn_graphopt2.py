#!/usr/bin/env python3
"""THE MISSING EXPERIMENT (fork-isolated): run the graph-opt (output-set)
bisection on the XNNPACK corpus IN-PROCESS. Each job runs in a forked child so
a native abort (the corpus is full of them) cannot kill the run.
usage: xnn_graphopt2.py <backend> <corpus> <n_jobs> <out.tsv>
"""
import os, sys, warnings, logging, random, json, signal
BK, CO, N, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
os.environ["MOBILE_BACKENDS"] = BK
sys.path.insert(0,"/data/jwen929")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from pathlib import Path
from mobile.gen.export import build_job
from mobile.executor.et_runner import run_pte
from mobile.executor import compare as cmp
from mobile.gen.diff.cone_predicate import output_set_localizer

def run_src(src, reps=1):
    try: j = build_job(src, BK, False)
    except Exception as e: return (f"BUILDEXC:{type(e).__name__}", None)
    if j.status != "READY": return (f"BUILD:{j.status}", None)
    allv = []
    for _ in range(reps):
        try: out = run_pte(j.pte, j.inputs)
        except Exception as e: return (f"RUNEXC:{type(e).__name__}", None)
        et = list(out)
        if len(et) > len(j.eager): et = et[len(et)-len(j.eager):]
        if len(et) < len(j.eager): return ("SHORT_OUT", None)
        allv.append([cmp._cmp(e,d,i)[0] for i,(e,d) in enumerate(zip(j.eager, et))])
    merged = ["MISMATCH" if any(v[i]=="MISMATCH" for v in allv) else allv[0][i]
              for i in range(len(allv[0]))]
    return ("RAN", merged)

def work(f):
    """runs in the child; returns list of tsv lines"""
    lines = []
    src0 = f.read_text()
    s, per = run_src(src0, reps=3)
    if s != "RAN":
        return [f"{f.name}\t-\tFULL_{s}\t-\t-\t-"]
    bad = [i for i,v in enumerate(per) if v == "MISMATCH"]
    if not bad:
        return [f"{f.name}\t-\tFULL_OK\t-\t-\t-"]
    for oi in bad:
        try:
            others, build_out, target, ret = output_set_localizer(str(f), oi)
        except Exception as e:
            lines.append(f"{f.name}\t{oi}\tLOCEXC:{type(e).__name__}\t-\t-\t-"); continue
        sA, perA = run_src(build_out([target]), reps=3)
        if sA != "RAN":
            lines.append(f"{f.name}\t{oi}\tA_{sA}\t{target}\t-\t-"); continue
        sB, perB = run_src(build_out(ret), reps=3)
        if sB != "RAN":
            lines.append(f"{f.name}\t{oi}\tB_{sB}\t{target}\t-\t-"); continue
        posB = ret.index(target)
        a, b = perA[0], perB[posB]
        v = "GRAPHOPT" if (a=="OK" and b=="MISMATCH") else ("OPERATOR" if a!="OK" else "NOREPRO")
        lines.append(f"{f.name}\t{oi}\t{v}\t{target}\tA={a}\tB={b}")
    return lines

files = sorted(Path(CO).glob("w*/*.py"))
random.seed(99); random.shuffle(files)
fh = open(OUT,"w")
from collections import Counter
ctr = Counter()
done = 0
for f in files:
    if done >= N: break
    done += 1
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            out = work(f)
        except BaseException as e:
            out = [f"{f.name}\t-\tPYEXC:{type(e).__name__}\t-\t-\t-"]
        try:
            os.write(w, ("\n".join(out)+"\n").encode())
        finally:
            os._exit(0)
    os.close(w)
    buf = b""
    with os.fdopen(r, "rb") as rf:
        buf = rf.read()
    _, status = os.waitpid(pid, 0)
    if not buf:
        line = f"{f.name}\t-\tNATIVE_ABORT_sig{os.WTERMSIG(status) if os.WIFSIGNALED(status) else '?'}\t-\t-\t-"
        buf = (line+"\n").encode()
    for line in buf.decode(errors="replace").splitlines():
        if not line.strip(): continue
        fh.write(line+"\n"); ctr[line.split("\t")[2].split(":")[0]] += 1
    fh.flush()
    if done % 50 == 0:
        print(done, dict(ctr), flush=True, file=sys.stderr)
print("FINAL", done, dict(ctr), file=sys.stderr)
fh.write("#FINAL "+json.dumps(dict(ctr))+"\n")
