#!/usr/bin/env python3
"""determinism.py — step 3b gate for analysis_single.md. Re-run each delegated failing job
(MISMATCH/CRASH on the VGF delegate, ops>=1) N times on the VGF runner and classify:
  REPRO   = all N runs reproduce the same non-OK verdict  → keep as a bug
  INTERMITTENT = 1..N-1 runs fail                          → report as intermittent
  FLAKY   = 0 runs fail (all OK on re-run)                 → drop

Replicates the worker+feeder path exactly: run vgf_runner.sh → reconstruct outputs from the
job's out_metas/user_pos → compare.select → compare.compare against the stored eager oracle.
Groups by operator and samples up to MAX_PER_OP distinct failing job_ids per op. Parallel.

Run: PYTHONPATH=/data/jwen929 .venv/bin/python findings/vgf_single/determinism.py [N] [MAX_PER_OP]
"""
from __future__ import annotations
import sys, json, subprocess, tempfile, collections
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/data/jwen929/mobile")
sys.path.insert(0, "/data/jwen929")
from mobile.executor import corpus as C           # noqa: E402
from mobile.executor import protocol as P         # noqa: E402
from mobile.executor import compare as cmp        # noqa: E402
import torch                                 # noqa: E402

OUT = ROOT / "findings/vgf_single"
RUNNER = ROOT / "vgf_client/vgf_runner.sh"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MAX_PER_OP = int(sys.argv[2]) if len(sys.argv) > 2 else 6
WORKERS = 24


def _rebuild(work: Path, out_metas, user_pos):
    files = sorted(work.glob("out_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
    n = len(files)
    pos = user_pos if user_pos is not None else list(range(len(out_metas)))
    outs = [torch.empty(0)] * max(n, (max(pos) + 1 if pos else 0))
    for k, pi in enumerate(pos):
        if k < len(out_metas):
            outs[pi] = P.tensor_from_meta_blob(out_metas[k], (work / f"out_{pi}.bin").read_bytes())
    return outs


def _one_run(pte: bytes, inputs, out_metas, user_pos, eager):
    """One runner invocation → (verdict, detail). Verdict in OK/MISMATCH/CRASH/SKIP."""
    with tempfile.TemporaryDirectory(prefix="vgf_det_") as d:
        work = Path(d); (work / "m.pte").write_bytes(pte)
        cmd = [str(RUNNER), "--pte", str(work / "m.pte"), "--out", str(work)]
        for i, t in enumerate(inputs):
            _, raw = P.tensor_to_meta_blob(t.contiguous())
            (work / f"i{i}").write_bytes(raw); cmd += ["--input", str(work / f"i{i}")]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            return "CRASH", "timeout"
        if p.returncode != 0:
            return ("SKIP" if p.returncode in (2, 3) else "CRASH"), f"rc={p.returncode}"
        try:
            outs = _rebuild(work, out_metas, user_pos)
        except Exception as e:
            return "SKIP", f"decode {e}"
        et = cmp.select(outs, user_pos)
        try:
            return cmp.compare(eager, et)
        except Exception as e:
            return "SKIP", f"compare {e}"


def gate_job(job_id: str):
    """Re-run one job N times → (job_id, op-less) reproduction record."""
    try:
        w, idx = job_id.split(":")
        path = C.job_file(str(ROOT / "corpus_v3/vgf"), job_id)
        frames = C.read_job(path)
        _jid, pte, inputs = P.decode_job(P.job_frames_from_pushjob(frames))
        eager, user_pos = P.eager_from_pushjob(frames)
        out_metas, _up = P.job_output_info(P.job_frames_from_pushjob(frames))
    except Exception as e:
        return {"job": job_id, "error": f"load {e}"}
    verdicts = []
    for _ in range(N):
        s, det = _one_run(pte, inputs, out_metas, user_pos, eager)
        verdicts.append((s, det))
    non_ok = [v for v in verdicts if v[0] != "OK"]
    kinds = collections.Counter(v[0] for v in verdicts)
    if len(non_ok) == N:
        cls = "REPRO"
    elif len(non_ok) == 0:
        cls = "FLAKY"
    else:
        cls = "INTERMITTENT"
    # representative detail = first non-OK detail (or OK)
    detail = non_ok[0][1] if non_ok else ""
    return {"job": job_id, "class": cls, "n": N, "fail": len(non_ok),
            "kinds": dict(kinds), "detail": detail}


def main():
    buckets = json.loads((OUT / "buckets.json").read_text())
    # step 3a: only delegated failures are candidate VGF bugs
    cand = []
    for bkey in ("mismatch_delegated", "crash_delegated"):
        by_op = collections.defaultdict(list)
        for op, job, reason, d in buckets.get(bkey, []):
            by_op[op].append(job)
        for op, jobs in by_op.items():
            for j in jobs[:MAX_PER_OP]:
                cand.append((op, bkey, j))
    print(f"determinism gate: {len(cand)} sampled failing jobs "
          f"({len(set(c[0] for c in cand))} ops), N={N}× each", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(gate_job, j): (op, bkey, j) for op, bkey, j in cand}
        done = 0
        for fut in as_completed(futs):
            op, bkey, j = futs[fut]
            r = fut.result(); r["op"] = op; r["bucket"] = bkey
            results.append(r)
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(cand)} gated", flush=True)

    (OUT / "determinism.json").write_text(json.dumps(results, indent=1))

    # per-op rollup
    op_cls = collections.defaultdict(lambda: collections.Counter())
    for r in results:
        if "class" in r:
            op_cls[(r["op"], r["bucket"])][r["class"]] += 1
    print("\n=== determinism-gated per op (REPRO = confirmed, survives N re-runs) ===")
    print(f"{'operator':32s} {'bucket':20s} REPRO INTERM FLAKY")
    for (op, bkey), c in sorted(op_cls.items(), key=lambda kv: -kv[1]['REPRO']):
        print(f"{op:32s} {bkey:20s} {c['REPRO']:5d} {c['INTERMITTENT']:6d} {c['FLAKY']:5d}")
    nrepro = sum(1 for r in results if r.get("class") == "REPRO")
    print(f"\nREPRO jobs: {nrepro}/{len(results)}  → confirmed-bug candidates")


if __name__ == "__main__":
    main()
