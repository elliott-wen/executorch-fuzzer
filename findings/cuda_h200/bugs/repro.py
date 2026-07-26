#!/usr/bin/env python3
"""repro.py — self-contained one-op repro for a CUDA-backend finding.

Rebuilds the exact corpus graph (its own seeded LEAVES) with build_job(src, "cuda") — the same
call the corpus was generated with — runs the .pte on an H200 via the ExecuTorch runtime, and
prints eager (CPU reference) vs device. For SKIP/CRASH forms it prints the device status/error.

Run in .venv-cuda with cuda-13.0 on PATH:
  CUDA_HOME=/usr/local/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64 \
    PYTHONPATH=/data/jwen929 /data/jwen929/mobile/.venv-cuda/bin/python repro.py <job_id>

<job_id> like w0:196. Defaults to the job baked into each bugs/<op>.md.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/data/jwen929")
import torch
from mobile.gen.export.job import build_job, load_functional_source

CORPUS = "/data/jwen929/mobile/corpus_v3/cuda"

def main():
    jid = sys.argv[1] if len(sys.argv) > 1 else "w0:196"
    prod, idx = jid.split(":")
    src = open(f"{CORPUS}/{prod}/{prod}_{idx}.py").read()
    graph_line = next(l for l in src.splitlines() if l.startswith("Graph:"))
    print(f"job {jid}: {graph_line}")
    g, leaves = load_functional_source(src)
    print("inputs:", [(tuple(t.shape), str(t.dtype)) for t in leaves])
    # eager reference
    try:
        with torch.no_grad():
            eager = g(*[t.clone() for t in leaves])
        eager = list(eager) if isinstance(eager, (tuple, list)) else [eager]
    except Exception as e:
        print("eager raised:", e); return
    # lower + run on device
    j = build_job(src, "cuda")
    print("lower:", j.status, j.detail[:120])
    if j.status != "READY":
        return
    from mobile.net.et_runner import run_pte
    try:
        out = run_pte(j.pte, j.inputs)
    except Exception as e:
        print("DEVICE STATUS: CRASH/SKIP —", str(e)[:200]); return
    up = j.user_pos or list(range(len(out)))
    print("all device outputs:", [(tuple(o.shape), str(o.dtype)) for o in out])
    for k, i in enumerate(up):
        e = eager[k] if k < len(eager) else None
        d = out[i]
        print(f"  user_output[{k}] (device idx {i}):")
        print(f"    eager : shape={tuple(e.shape) if e is not None else '?'} "
              f"vals={(e.flatten()[:8].tolist() if e is not None else '?')}")
        print(f"    device: shape={tuple(d.shape)} vals={d.detach().cpu().flatten()[:8].tolist()}")

if __name__ == "__main__":
    main()
