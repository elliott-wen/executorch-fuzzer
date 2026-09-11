"""For each job_id on stdin, run it on the host runner and report the runtime's own reason."""
import sys, subprocess, tempfile, re, shutil
sys.path.insert(0, "/data/jwen929")
from pathlib import Path
from mobile.executor import corpus as C, protocol as P

CORPUS = "/data/jwen929/mobile/corpus_v5/cadence"
RUNNER = "/data/jwen929/mobile/cadence_client/cadence_runner.sh"
MISS = re.compile(r"kernel '([^']+)' not found")
for line in sys.stdin:
    jid = line.strip()
    if not jid: continue
    try:
        _, pte, inputs = P.decode_job(C.read_job(C.job_file(CORPUS, jid)))
    except Exception as e:
        print(f"{jid}\tDECODE_ERR\t{e}"); continue
    d = Path(tempfile.mkdtemp(prefix="cs_"))
    try:
        (d/"m.pte").write_bytes(pte)
        ins = []
        for i, t in enumerate(inputs):
            (d/f"i{i}.bin").write_bytes(t.contiguous().numpy().tobytes()); ins += ["--input", str(d/f"i{i}.bin")]
        r = subprocess.run([RUNNER, "--pte", str(d/"m.pte"), "--out", str(d/"o"), *ins],
                           capture_output=True, text=True, timeout=120)
        log = (d/"o"/"run.log")
        txt = log.read_text() if log.exists() else ""
        m = MISS.search(txt)
        if m:
            print(f"{jid}\tMISSING_KERNEL\t{m.group(1)}")
        else:
            first = next((l for l in txt.splitlines() if l.startswith("E ")), "")
            print(f"{jid}\trc={r.returncode}\t{first[:160]}")
    except Exception as e:
        print(f"{jid}\tRUN_ERR\t{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(d, ignore_errors=True)
