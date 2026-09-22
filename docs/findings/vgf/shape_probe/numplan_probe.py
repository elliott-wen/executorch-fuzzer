#!/usr/bin/env python3
"""For NumPlan candidates: print eager vs device values, relative error, and the A-side
(baked-cone) result, so the 'context-dependent numeric treatment' claim can be checked
against fp16 rounding magnitude."""
import os, sys, subprocess, tempfile, pathlib
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from mobile.executor import corpus as C, protocol as P, compare as cmp
from mobile.gen.export import build_job
from mobile.gen.diff.cone_predicate import cone_localizer

CO = "/data/jwen929/mobile/corpus_v4/vgf"
RUNNER_SH = "/data/jwen929/mobile/vgf_client/vgf_runner.sh"
CASES = [
    ("w106:621", 0, "clamp", ["n2"]), ("w107:319", 1, "log", ["n1","n7","n3"]),
    ("w114:256", 0, "min", ["n3","n0","n2"]), ("w120:300", 2, "squeeze_copy", ["n0","n1","n5"]),
    ("w121:253", 1, "reciprocal", ["n2"]), ("w22:81", 1, "view_copy", ["n0","n3"]),
]

def dev_out(j, tag):
    d = pathlib.Path(tempfile.mkdtemp(prefix=f"np_{tag}_"))
    pte = d / "m.pte"; pte.write_bytes(j.pte)
    cmd = [RUNNER_SH, "--pte", str(pte), "--out", str(d)]
    for i, t in enumerate(j.inputs):
        _, raw = P.tensor_to_meta_blob(t.contiguous())
        p = d / f"in_{i}.bin"; p.write_bytes(raw); cmd += ["--input", str(p)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0: return None
    outs = sorted(d.glob("out_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
    e0 = j.eager[0]; up = j.user_pos
    idx = up[0] if (up and len(up) == 1 and up[0] < len(outs)) else len(outs) - 1
    t = torch.frombuffer(bytearray(outs[idx].read_bytes()), dtype=e0.dtype)
    return t.reshape(e0.shape) if t.numel() == e0.numel() else None

for job_id, out_idx, tgt, needed in CASES:
    anc, build_src, target = cone_localizer(C.job_file(CO, job_id, "py"), out_idx)
    row = {}
    for label, kept in (("A_baked", []), ("B_live", needed)):
        j = build_job(build_src(kept), "vgf", False)
        if j.status != "READY": row[label] = f"BUILD:{j.status}"; continue
        d = dev_out(j, job_id.replace(":", "_"))
        if d is None: row[label] = "RUN_FAIL"; continue
        e = j.eager[0]
        ef, df = e.flatten().float(), d.flatten().float()
        st, _ = cmp._cmp(e, d, 0)
        denom = ef.abs().clamp_min(1e-12)
        rel = float(((df - ef).abs() / denom).max())
        m = ef.abs() > 1e-3
        ratio = (df[m] / ef[m]) if m.any() else torch.tensor([])
        rstr = (f" ratio_mean={float(ratio.mean()):.4f} ratio_std={float(ratio.std() if ratio.numel()>1 else torch.tensor(0.)):.4f}"
                if ratio.numel() else "")
        row[label] = (f"{st} rel_err={rel:.4g}{rstr}\n        eager={ef[:4].tolist()}\n        device={df[:4].tolist()}")
    print(f"\n=== {job_id} out{out_idx} target={tgt} needed_live={needed}")
    for k, v in row.items(): print(f"  {k}: {v}")
