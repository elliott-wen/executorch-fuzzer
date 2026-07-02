#!/usr/bin/env python3
"""Replay: aten::select_scatter — to_edge decomposition mis-promotes output dtype.

Two self-contained demonstrations:
  (A) Corpus job w0:1160 — out[4] eager dtype=int64 vs portable dtype=float32.
  (B) Isolated export proof — select_scatter(int64 self, float32 src):
      eager out dtype=int64 vs to_edge terminal node dtype=float32 (root cause:
      decomposition to torch.where promotes result_type(src, self)).

Usage: cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_select_scatter-dtype.py
(filter noise: | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference")
"""
import importlib.util as u
import os
import sys
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")  # so `import mobile` resolves
warnings.filterwarnings("ignore")

import torch
from torch.export import export
from executorch.exir import to_edge
from mobile.net import corpus, protocol, et_runner

REPO = "/data/jwen929/mobile"
JOB = "corpus/portable/w0/w0_1160.py"   # n8 = select_scatter(int64, float32 src)
OUT = 4                                  # offending USER_OUTPUT index


def load(job_py):
    job = os.path.join(REPO, job_py.replace(".py", ".job"))
    frames = corpus.read_job(job)
    job_id, pte, inputs = protocol.decode_job(protocol.job_frames_from_pushjob(frames))
    return job_id, pte, inputs


def eager(job_py, inputs):
    s = u.spec_from_file_location("m", os.path.join(REPO, job_py))
    m = u.module_from_spec(s)
    s.loader.exec_module(m)
    outs = m.g(*[t.clone() for t in inputs])
    return [x for x in outs if hasattr(x, "dtype")]


def corpus_replay():
    print("=== (A) Corpus job w0:1160 — output dtype divergence ===")
    job_id, pte, inputs = load(JOB)
    pouts = et_runner.run_pte(pte, inputs)
    eouts = eager(JOB, inputs)
    p, e = pouts[OUT], eouts[OUT]
    print(f"job {job_id} out[{OUT}]: portable dtype={p.dtype}  eager dtype={e.dtype}")
    reproduced = p.dtype != e.dtype
    print("  CORPUS: REPRODUCED (dtype differs)" if reproduced
          else "  CORPUS: dtypes match")
    return reproduced


def isolated_proof():
    print("=== (B) Isolated export proof — select_scatter(int64, float32 src) ===")

    class M(torch.nn.Module):
        def forward(self, x, s):
            return torch.select_scatter(x, s, 0, 1)

    x = torch.zeros(3, 2, dtype=torch.int64)
    s = torch.ones(2, dtype=torch.float32)
    eager_dt = M()(x, s).dtype
    edge = to_edge(export(M().eval(), (x, s)))
    out = list(edge.exported_program().graph.nodes)[-1].args[0][0]
    edge_dt = out.meta["val"].dtype
    print(f"eager select_scatter dtype: {eager_dt}")
    print(f"edge terminal node '{out.name}' dtype: {edge_dt}")
    reproduced = eager_dt == torch.int64 and edge_dt == torch.float32
    print("  ISOLATED: REPRODUCED (int64 -> float32 via where promotion)"
          if reproduced else "  ISOLATED: no divergence")
    return reproduced


def main():
    a = corpus_replay()
    print()
    b = isolated_proof()
    print()
    ok = a or b
    print("REPLAY: bug REPRODUCED" if ok else "REPLAY: no divergence")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
