#!/usr/bin/env python3
"""Scan corpus_v3/ethos-u .job headers -> manifest.tsv: job_id, op, delegated_ops, non, calls, desc.

Every job is one operator (single-op corpus). The .job file begins with a small binary
length prefix then a JSON header: {"job_id":..,"desc":"n0*=OP(..)","delegated":{"ops":d,..}}.
We read only the first block of each file and extract the first balanced {...} JSON object.
"""
import json, re, sys
from pathlib import Path

CORPUS = Path("/data/jwen929/mobile/corpus_v3/ethos-u")
OUT = Path("/data/jwen929/mobile/findings/ethos_u/_work/manifest.tsv")

op_re = re.compile(r"=([a-zA-Z0-9_.]+)\(")

def first_json(b: bytes):
    i = b.find(b"{")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(b)):
        c = b[j]
        if c == 0x7b:  # {
            depth += 1
        elif c == 0x7d:  # }
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

n = 0
bad = 0
with OUT.open("w") as out:
    out.write("job_id\top\tdelegated_ops\tnon\tcalls\tdesc\n")
    for jf in sorted(CORPUS.glob("w*/*.job")):
        try:
            with jf.open("rb") as fh:
                head = fh.read(2048)
        except Exception:
            bad += 1
            continue
        hdr = first_json(head)
        if not hdr:
            bad += 1
            continue
        jid = hdr.get("job_id", jf.stem)
        desc = hdr.get("desc", "")
        dele = hdr.get("delegated", {}) or {}
        ops = dele.get("ops", -1)
        non = dele.get("non", -1)
        calls = dele.get("calls", -1)
        out.write(f"{jid}\t{op_of(desc)}\t{ops}\t{non}\t{calls}\t{desc}\n")
        n += 1
        if n % 10000 == 0:
            print(f"  {n} ...", flush=True)
print(f"done: {n} jobs, {bad} unparseable -> {OUT}", flush=True)
