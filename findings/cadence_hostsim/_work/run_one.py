"""Extract one corpus job's .pte + inputs to a dir and run cadence_runner.sh on it, showing run.log."""
import sys, subprocess, tempfile, os
sys.path.insert(0, "/data/jwen929")
from pathlib import Path
from mobile.net import corpus as C, protocol as P
import torch

CORPUS = "/data/jwen929/mobile/corpus_v5/cadence"
jid = sys.argv[1]
frames = C.read_job(C.job_file(CORPUS, jid))
job_id, pte, inputs = P.decode_job(frames)
d = Path(tempfile.mkdtemp(prefix="cadrun_"))
(d / "m.pte").write_bytes(pte)
ins = []
for i, t in enumerate(inputs):
    p = d / f"in_{i}.bin"
    p.write_bytes(t.contiguous().numpy().tobytes())
    ins += ["--input", str(p)]
cmd = ["/data/jwen929/mobile/cadence_client/cadence_runner.sh",
       "--pte", str(d / "m.pte"), "--out", str(d / "out"), *ins]
r = subprocess.run(cmd, capture_output=True, text=True)
print(f"job {jid}  rc={r.returncode}")
print("--- stderr:", r.stderr.strip()[:500])
log = d / "out" / "run.log"
if log.exists():
    print("--- run.log:"); print(log.read_text()[:2000])
print("workdir:", d)
