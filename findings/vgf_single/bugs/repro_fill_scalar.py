#!/usr/bin/env python3
"""repro_fill_scalar.py — VGF delegate: fill.Scalar emits ZEROS, ignoring the fill scalar.

Confirmed VGF-delegate WRONG-VALUE (ZEROED) bug. A one-op graph `aten.fill.Scalar(L0, c)`
returns a tensor shaped like L0 filled with constant `c`. Delegated to VGF it returns
all-zeros regardless of `c` — the scalar operand is dropped. Deterministic (5/5), fully
delegated (ops=1, non_delegated=0 → no portable path, so the VGF delegate genuinely ran).

Replays the exact corpus jobs on the built VGF runner and prints eager-vs-device. Requires
the runner (mobile/vgf_client/build_runner.sh) + emulation layer. No AoT env needed (replays
the stored .pte).

Run: PYTHONPATH=/data/jwen929 .venv/bin/python findings/vgf_single/bugs/repro_fill_scalar.py
"""
import subprocess, tempfile, sys
from pathlib import Path
sys.path.insert(0, "/data/jwen929")
import torch
from mobile.executor import corpus as C, protocol as P, compare as cmp

ROOT = Path("/data/jwen929/mobile"); RUNNER = ROOT / "vgf_client/vgf_runner.sh"
# corpus jobs: fill.Scalar(L0, c) for various c — see findings/vgf_single/skiplog.tsv
JOBS = ["w0:103", "w0:565", "w1:433"]  # fill 6, fill -4, fill -8

def replay(job):
    frames = C.read_job(C.job_file(str(ROOT / "corpus_v3/vgf"), job))
    _j, pte, inputs = P.decode_job(P.job_frames_from_pushjob(frames))
    eager, up = P.eager_from_pushjob(frames)
    om, _ = P.job_output_info(P.job_frames_from_pushjob(frames))
    with tempfile.TemporaryDirectory() as d:
        work = Path(d); (work / "m.pte").write_bytes(pte)
        cmd = [str(RUNNER), "--pte", str(work / "m.pte"), "--out", str(work)]
        for i, t in enumerate(inputs):
            _, raw = P.tensor_to_meta_blob(t.contiguous())
            (work / f"i{i}").write_bytes(raw); cmd += ["--input", str(work / f"i{i}")]
        subprocess.run(cmd, capture_output=True, timeout=90)
        files = sorted(work.glob("out_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
        pos = up if up is not None else list(range(len(om)))
        outs = [torch.empty(0)] * max(len(files), (max(pos) + 1 if pos else 0))
        for k, pi in enumerate(pos):
            if k < len(om): outs[pi] = P.tensor_from_meta_blob(om[k], (work / f"out_{pi}.bin").read_bytes())
        et = cmp.select(outs, up)
    return eager[0], (et[-1] if et else torch.empty(0))

for job in JOBS:
    ref, dev = replay(job)
    r = ref.flatten()[:4].tolist(); d = dev.flatten()[:4].tolist()
    bug = len(d) == len(r) and all(x == 0 for x in d) and any(x != 0 for x in r)
    print(f"{job}: eager={r}  device={d}  {'BUG: ZEROED (scalar dropped)' if bug else ''}")
