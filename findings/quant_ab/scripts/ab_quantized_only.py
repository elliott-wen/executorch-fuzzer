#!/usr/bin/env python3
"""The sharp comparison: restrict the paired A/B to graphs that were ACTUALLY quantized.

--quantize is a no-op for any op the backend's quantizer doesn't annotate, so most of the
int8 corpus is byte-identical work to the float corpus and dilutes every rate. A job really
got q/dq iff its int8 .pte has MORE nodes than its float .pte (ops + non_delegated), since
PT2E only ever adds quantize/dequantize nodes.

usage: ab_quantized_only.py <float_corpus> <int8_corpus> <float.skip.tsv> <int8.skip.tsv>
"""
import sys, re, csv, collections, pathlib
FC, QC, FSKIP, QSKIP = sys.argv[1:5]
RE_G = re.compile(r"Graph: (.*)"); RE_OP = re.compile(r"n\d+\*?=([A-Za-z_][\w.]*)\(")
RE_D = re.compile(r"# delegated: backend=\S+ ops=(\d+) non_delegated=(\d+)")

def scan(d):
    out = {}
    for p in pathlib.Path(d).glob("*/*.py"):
        h = p.read_text(errors="replace")[:700]
        g, m = RE_G.search(h), RE_D.search(h)
        if not (g and m): continue
        ops = RE_OP.findall(g.group(1))
        out[p.stem.replace("_", ":", 1)] = (ops[0] if len(ops) == 1 else "<multi>",
                                            int(m.group(1)), int(m.group(2)))
    return out

F, Q = scan(FC), scan(QC)
def load(p):
    return {r["job"]: r["status"] for r in csv.DictReader(open(p), delimiter="\t")}
FS, QS = load(FSKIP), load(QSKIP)

pair = sorted(set(F) & set(Q))
qtz = [j for j in pair if (Q[j][1] + Q[j][2]) > (F[j][1] + F[j][2])]
print(f"paired graphs {len(pair)}   ACTUALLY quantized (int8 .pte has extra q/dq nodes): "
      f"{len(qtz)} ({100*len(qtz)/len(pair):.1f}%)")

for label, sel in (("ALL paired", pair), ("quantized subset", qtz), ("untouched subset", [j for j in pair if j not in set(qtz)])):
    fv = collections.Counter(FS.get(j, "OK") for j in sel)
    qv = collections.Counter(QS.get(j, "OK") for j in sel)
    n = len(sel) or 1
    print(f"\n{label}  (n={len(sel)})")
    print(f"  float: OK {fv['OK']:5d} ({100*fv['OK']/n:5.1f}%)  MISM {fv['MISMATCH']:4d}  SKIP {fv['SKIP']:4d}  CRASH {fv['CRASH']:3d}")
    print(f"  int8 : OK {qv['OK']:5d} ({100*qv['OK']/n:5.1f}%)  MISM {qv['MISMATCH']:4d}  SKIP {qv['SKIP']:4d}  CRASH {qv['CRASH']:3d}")

# delegation shift on the quantized subset
dl_f = sum(1 for j in qtz if F[j][1] >= 1); dl_q = sum(1 for j in qtz if Q[j][1] >= 1)
print(f"\ndelegated (ops>=1) on the quantized subset: float {dl_f}/{len(qtz)}  int8 {dl_q}/{len(qtz)}")
print("\nops in the quantized subset (count, float OK% -> int8 OK%):")
byop = collections.defaultdict(list)
for j in qtz: byop[Q[j][0]].append(j)
for op, js in sorted(byop.items(), key=lambda kv: -len(kv[1])):
    fo = sum(1 for j in js if FS.get(j, "OK") == "OK"); qo = sum(1 for j in js if QS.get(j, "OK") == "OK")
    flag = "  <<<" if qo < fo else ""
    print(f"  {op:32s} n={len(js):4d}   {100*fo/len(js):5.1f}% -> {100*qo/len(js):5.1f}%{flag}")
