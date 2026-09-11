"""For each job_id, decode its .pte from the corpus .job and report which operator names it
carries — specifically whether ANY cadence:: kernel is called (the pass-based analog of
delegated.ops>=1, cf. cortex-m filter-3a adaptation)."""
import sys, re, gzip
sys.path.insert(0, "/data/jwen929")
from mobile.executor import corpus as C, protocol as P

CORPUS = "/data/jwen929/mobile/corpus_v5/cadence"
NAME = re.compile(rb"[a-z_0-9]+::[a-z_0-9]+(?:\.[a-zA-Z_0-9]+)?")

def ops_of(job_id):
    frames = C.read_job(C.job_file(CORPUS, job_id))
    _, pte, _ = P.decode_job(frames)
    names = set(m.decode() for m in NAME.findall(pte))
    return names

if __name__ == "__main__":
    for line in sys.stdin:
        jid = line.strip()
        if not jid:
            continue
        try:
            names = ops_of(jid)
        except Exception as e:
            print(f"{jid}\tERR\t{type(e).__name__}: {e}")
            continue
        cad = sorted(n for n in names if n.startswith("cadence::"))
        print(f"{jid}\t{'CADENCE' if cad else 'portable'}\t{','.join(cad)}")
