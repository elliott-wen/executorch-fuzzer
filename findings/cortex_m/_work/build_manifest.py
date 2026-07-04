#!/usr/bin/env python3
"""Scan corpus_v3/cortex-m .job headers -> manifest.tsv: job_id, op, delegated_ops, desc.

cortex-m is PASS-BASED, not a delegate: the .job header's delegated.ops is structurally
ALWAYS 0 (no partitioner ran), so that column is a dead signal here — kept only for schema
parity with the ethos-u pipeline. The real "which kernel ran" signal (which cortex_m::
CMSIS-NN ops the lowered program calls) is computed per-job by re-lowering the non-OK
worklist (see cm_kernels.py), not from this header.
"""
import json, re
from pathlib import Path

CORPUS = Path("/data/jwen929/mobile/corpus_v3/cortex-m")
OUT = Path("/data/jwen929/mobile/findings/cortex_m/_work/manifest.tsv")
op_re = re.compile(r"=([a-zA-Z0-9_.]+)\(")

def first_json(b: bytes):
    i = b.find(b"{")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(b)):
        c = b[j]
        if c == 0x7b:
            depth += 1
        elif c == 0x7d:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(b[i:j+1].decode("utf-8", "replace"))
                except Exception:
                    return None
    return None

def op_of(desc: str) -> str:
    m = op_re.search(desc or "")
    return m.group(1) if m else "?"

n = bad = 0
with OUT.open("w") as out:
    out.write("job_id\top\tdelegated_ops\tnon\tcalls\tdesc\n")
    for jf in sorted(CORPUS.glob("w*/*.job")):
        try:
            with jf.open("rb") as fh:
                head = fh.read(2048)
        except Exception:
            bad += 1; continue
        hdr = first_json(head)
        if not hdr:
            bad += 1; continue
        jid = hdr.get("job_id", jf.stem)
        desc = hdr.get("desc", "")
        dele = hdr.get("delegated", {}) or {}
        out.write(f"{jid}\t{op_of(desc)}\t{dele.get('ops',-1)}\t{dele.get('non',-1)}\t{dele.get('calls',-1)}\t{desc}\n")
        n += 1
        if n % 10000 == 0:
            print(f"  {n} ...", flush=True)
print(f"done: {n} jobs, {bad} unparseable -> {OUT}", flush=True)
