"""Scan one corpus shard: for every .job, list the cadence:: ops its .pte calls.
out: <job_id>\t<CADENCE|portable>\t<comma-separated cadence ops>"""
import sys, re
sys.path.insert(0, "/data/jwen929")
from pathlib import Path
from mobile.executor import corpus as C, protocol as P

NAME = re.compile(rb"cadence::[a-z_0-9]+(?:\.[a-zA-Z_0-9]+)?")
shard = Path(sys.argv[1])
out = open(sys.argv[2], "w")
for jf in sorted(shard.glob("*.job")):
    jid = jf.stem.replace("_", ":", 1)
    try:
        _, pte, _ = P.decode_job(C.read_job(jf))
    except Exception as e:
        out.write(f"{jid}\tERR\t{type(e).__name__}\n"); continue
    cad = sorted(set(m.decode() for m in NAME.findall(pte)))
    out.write(f"{jid}\t{'CADENCE' if cad else 'portable'}\t{','.join(cad)}\n")
out.close()
