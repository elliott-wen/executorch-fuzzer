#!/usr/bin/env python3
"""Coverage ledger (analysis_single.md step 2/coverage gate):
scheduled (EXECUTORCH_OPS=228) vs produced (in corpus) vs delegated/portable, plus
generation-time attrition from corpus_v3/ethos-u/_crashes/*.py.

  scheduled = produced_delegated + produced_portable + never_produced
"""
import re, collections
from pathlib import Path
import mobile.gen.ops.allowlist as A

CORPUS = Path("/data/jwen929/mobile/corpus_v3/ethos-u")
MAN = Path("/data/jwen929/mobile/findings/ethos_u/_work/manifest.tsv")

scheduled = set(A.EXECUTORCH_OPS)

produced_del = set()
produced_port = set()
op_del = collections.Counter()
op_port = collections.Counter()
for ln in MAN.read_text().splitlines()[1:]:
    f = ln.split("\t")
    if len(f) < 3:
        continue
    op, d = f[1], f[2]
    try:
        d = int(d)
    except ValueError:
        continue
    if d >= 1:
        produced_del.add(op); op_del[op] += 1
    else:
        produced_port.add(op); op_port[op] += 1

produced = produced_del | produced_port
never = scheduled - produced
# ops produced but NOT in the scheduled allowlist (decomposition edge-op names, aliases)
extra = produced - scheduled

# generation-time attrition: which ops crashed at build time
op_re = re.compile(r"Graph:\s*n0\*?=([a-zA-Z0-9_.]+)\(")
gen_crash_ops = collections.Counter()
for py in CORPUS.glob("_crashes/*.py"):
    try:
        txt = py.read_text(errors="replace")
    except Exception:
        continue
    m = op_re.search(txt)
    if m:
        gen_crash_ops[m.group(1)] += 1

print(f"SCHEDULED (EXECUTORCH_OPS): {len(scheduled)}")
print(f"PRODUCED (in corpus):       {len(produced)}  "
      f"[delegated-capable {len(produced_del)}, always-portable {len(produced_port & (produced - produced_del))}]")
print(f"NEVER PRODUCED (gap):       {len(never)}")
print(f"PRODUCED-but-not-scheduled (decomp/alias names): {len(extra)}")
print()
print(f"ASSERT scheduled == produced_in_schedule + never : "
      f"{len(scheduled & produced)} + {len(never)} = {len(scheduled & produced)+len(never)} vs {len(scheduled)}")
print()
print("== NEVER-PRODUCED ops (coverage gaps — scheduled but never lowered) ==")
for op in sorted(never):
    gc = gen_crash_ops.get(op, 0)
    tag = f"gen-crash x{gc}" if gc else "no job produced (seed-infeasible / partition/lowering attrition)"
    print(f"  {op:<40} {tag}")
print()
print(f"== generation-time _crashes: {sum(gen_crash_ops.values())} graphs over {len(gen_crash_ops)} ops ==")
for op, c in gen_crash_ops.most_common(30):
    inproduced = "PRODUCED-elsewhere" if op in produced else "NEVER-produced"
    print(f"  {op:<40} x{c:<5} {inproduced}")
print()
print("== ALWAYS-PORTABLE produced ops (ethos-u never delegated them; 0 delegated jobs) ==")
always_port = sorted(produced_port - produced_del)
print(f"  ({len(always_port)} ops) " + ", ".join(always_port))
print()
print("== DELEGATED-CAPABLE ops (>=1 delegated job — the ethos-u backend surface) ==")
print(f"  ({len(produced_del)} ops) " + ", ".join(sorted(produced_del)))
