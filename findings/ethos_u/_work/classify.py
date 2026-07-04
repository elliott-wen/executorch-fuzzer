#!/usr/bin/env python3
"""Classify each DELEGATED non-OK job by input domain (filters 3d/3e of analysis_single.md).

For every non-OK row in skip.tsv whose job is delegated (ops>=1), load the graph's LEAVES
and bucket:
  NONFINITE-INPUT : a float leaf contains nan/inf (domain-fuzz injection) -> 3e exception,
                    the divergence is non-finite-input handling, not a clean finite-domain kernel bug.
  INT-SHIFT-NEG   : a bitwise-shift op whose shift operand has negative entries -> UB, low-confidence.
  FINITE-CLEAN    : all leaves finite (and not the UB-shift case) -> genuine ethos-u candidate.

Prints a per-op summary and lists the FINITE-CLEAN candidates (the real worklist).
"""
import runpy, collections, sys
import torch

W = "/data/jwen929/mobile/findings/ethos_u/_work"
CORPUS = "/data/jwen929/mobile/corpus_v3/ethos-u"

man = {}
for ln in open(f"{W}/manifest.tsv").read().splitlines()[1:]:
    f = ln.split("\t")
    if len(f) >= 6:
        man[f[0]] = (f[1], f[2], f[5])   # op, delegated_ops, desc

def jid_to_py(jid):
    w, n = jid.split(":")
    return f"{CORPUS}/{w}/{w}_{n}.py"

def classify(jid, desc):
    try:
        ns = runpy.run_path(jid_to_py(jid))
        L = ns["LEAVES"]
    except Exception as e:
        return "LOADFAIL", str(e)[:60]
    nonfinite = False
    for t in L:
        if t.dtype.is_floating_point and not bool(torch.isfinite(t).all()):
            nonfinite = True
    if nonfinite:
        return "NONFINITE-INPUT", ""
    if "shift" in desc:
        # shift operand is the 2nd tensor leaf if present
        for t in L[1:]:
            if not t.dtype.is_floating_point and bool((t < 0).any()):
                return "INT-SHIFT-NEG", ""
    return "FINITE-CLEAN", ""

rows = []
for ln in open(f"{W}/skip.tsv").read().splitlines()[1:]:
    f = ln.split("\t")
    if len(f) < 2:
        continue
    st, jid = f[0], f[1]
    op, d, desc = man.get(jid, ("?", "-2", ""))
    if d in ("0", "-2"):     # portable / unknown -> not ethos-u
        continue
    rows.append((st, jid, op, desc, f[2] if len(f) > 2 else ""))

buckets = collections.Counter()
per_op = collections.defaultdict(lambda: collections.Counter())
clean = []
for st, jid, op, desc, reason in rows:
    cls, note = classify(jid, desc)
    buckets[cls] += 1
    per_op[op][cls] += 1
    if cls == "FINITE-CLEAN":
        clean.append((op, jid, st, desc, reason))

print(f"DELEGATED non-OK rows classified: {len(rows)}")
print(f"buckets: {dict(buckets)}")
print("\n== per-op classification (delegated non-OK) ==")
print(f"{'operator':<32}{'FINITE-CLEAN':>13}{'NONFINITE':>11}{'SHIFT-NEG':>11}{'other':>7}")
for op in sorted(per_op):
    c = per_op[op]
    other = sum(v for k, v in c.items() if k not in ("FINITE-CLEAN","NONFINITE-INPUT","INT-SHIFT-NEG"))
    print(f"{op:<32}{c['FINITE-CLEAN']:>13}{c['NONFINITE-INPUT']:>11}{c['INT-SHIFT-NEG']:>11}{other:>7}")

print(f"\n== FINITE-CLEAN candidates (genuine ethos-u worklist): {len(clean)} ==")
for op, jid, st, desc, reason in sorted(clean):
    print(f"  {op:<28} {jid:<10} {st:<9} {desc[:36]:<37} {reason[:50]}")
