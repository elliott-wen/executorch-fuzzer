#!/usr/bin/env python3
"""build_manifest.py — one pass over the corpus_v3/openvino .job headers to extract, per job:
   job_id, operator (from desc), delegated-op count (delegated.ops).
This is the coverage denominator + delegation source for analyze.py (steps 2/3a of
analysis_single.md). Reads only the leading JSON header of each .job (binary payload follows).

Run:  PYTHONPATH=/data/jwen929 .venv/bin/python findings/openvino_single/build_manifest.py
"""
from __future__ import annotations
import re, json
from pathlib import Path

ROOT = Path("/data/jwen929/mobile")
CORPUS = ROOT / "corpus_v3/openvino"
OUT = ROOT / "findings/openvino_single"
MANIFEST = OUT / "manifest.tsv"

OP_RE = re.compile(r"n0\*?=([A-Za-z0-9_.]+?)\(")

def op_of(desc: str) -> str:
    m = OP_RE.search(desc or "")
    return m.group(1) if m else "?"

def parse_header(path: Path):
    """Return (job_id, op, deleg_ops) from a .job's leading JSON header, or None."""
    raw = path.read_bytes()[:2048].decode("latin-1", "replace")
    i = raw.find('{"job_id"')
    if i < 0:
        return None
    # find the end of the JSON object by brace matching
    depth = 0
    end = -1
    for j in range(i, len(raw)):
        c = raw[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    if end < 0:
        return None
    try:
        obj = json.loads(raw[i:end])
    except Exception:
        return None
    jid = obj.get("job_id", "")
    op = op_of(obj.get("desc", ""))
    deleg = obj.get("delegated", {}) or {}
    ops = deleg.get("ops")
    return jid, op, ops

def main() -> int:
    n = 0
    with open(MANIFEST, "w") as f:
        f.write("job_id\toperator\tdeleg_ops\n")
        for wd in sorted(CORPUS.glob("w*")):
            if not wd.is_dir():
                continue
            for jp in wd.glob("*.job"):
                r = parse_header(jp)
                if r is None:
                    continue
                jid, op, ops = r
                f.write(f"{jid}\t{op}\t{ops if ops is not None else ''}\n")
                n += 1
    print(f"manifest: {n} jobs -> {MANIFEST}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
