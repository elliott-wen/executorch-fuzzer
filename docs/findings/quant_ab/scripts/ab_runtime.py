#!/usr/bin/env python3
"""Paired runtime A/B: same graph, float .pte vs int8 .pte, same local runtime.

Inputs: the paired corpus dir (defines the universe + the op of each job) and the two
feed skip-logs (which record ONLY non-OK jobs, so OK = universe - logged).

Emits the transition matrix (what float-OK became under int8 and vice versa) and the
per-op table for every op whose verdicts changed.

usage: ab_runtime.py <paired_float_corpus> <float.skip.tsv> <int8.skip.tsv> <out_prefix>
"""
import sys, re, csv, collections, pathlib, json

CORPUS, FSKIP, QSKIP, PREF = sys.argv[1:5]
RE_G = re.compile(r"Graph: (.*)"); RE_OP = re.compile(r"n\d+\*?=([A-Za-z_][\w.]*)\(")
RE_DEL = re.compile(r"# delegated: backend=(\S+) ops=(\d+)")

op_of, deleg_of = {}, {}
for p in pathlib.Path(CORPUS).glob("*/*.py"):
    head = p.read_text(errors="replace")[:700]
    g, d = RE_G.search(head), RE_DEL.search(head)
    if not g: continue
    ops = RE_OP.findall(g.group(1))
    jid = p.stem.replace("_", ":", 1)          # w7_412 -> w7:412
    op_of[jid] = ops[0] if len(ops) == 1 else "<multi>"
    deleg_of[jid] = int(d.group(2)) if d else -1

def load(path):
    st, rs = {}, {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            st[row["job"]] = row["status"]; rs[row["job"]] = row["reason"]
    return st, rs

FS, FR = load(FSKIP)
QS, QR = load(QSKIP)
univ = sorted(op_of)
fv = {j: FS.get(j, "OK") for j in univ}
qv = {j: QS.get(j, "OK") for j in univ}

trans = collections.Counter((fv[j], qv[j]) for j in univ)
tot = collections.Counter()
for j in univ: tot["f:" + fv[j]] += 1; tot["q:" + qv[j]] += 1

print(f"paired graphs: {len(univ)}\n")
sts = ["OK", "MISMATCH", "SKIP", "CRASH", "TIMEOUT"]
print(f"{'':12s}" + "".join(f"{s:>10s}" for s in sts) + f"{'total':>10s}")
for a in sts:
    row = [trans.get((a, b), 0) for b in sts]
    if sum(row) == 0 and tot['f:'+a] == 0: continue
    print(f"float {a:<7s}" + "".join(f"{v:10d}" for v in row) + f"{sum(row):10d}")
print(f"{'int8 total':12s}" + "".join(f"{tot['q:'+s]:10d}" for s in sts))

changed = [j for j in univ if fv[j] != qv[j]]
print(f"\nverdict CHANGED on {len(changed)} of {len(univ)} graphs ({100*len(changed)/len(univ):.1f}%)")

per_op = collections.defaultdict(lambda: collections.Counter())
for j in univ:
    per_op[op_of[j]][("f", fv[j])] += 1
    per_op[op_of[j]][("q", qv[j])] += 1
    if fv[j] != qv[j]: per_op[op_of[j]]["changed"] += 1

with open(PREF + "_per_op.tsv", "w") as fh:
    fh.write("op\tn\tf_OK\tf_MISMATCH\tf_SKIP\tf_CRASH\tq_OK\tq_MISMATCH\tq_SKIP\tq_CRASH\tchanged\n")
    for op in sorted(per_op):
        c = per_op[op]
        n = sum(c[("f", s)] for s in sts)
        fh.write("\t".join(str(x) for x in [op, n] +
                 [c[("f", s)] for s in sts[:4]] + [c[("q", s)] for s in sts[:4]] + [c["changed"]]) + "\n")

print("\nops with the largest verdict change (int8 worse ▲ / better ▼):")
rows = []
for op in per_op:
    c = per_op[op]
    n = sum(c[("f", s)] for s in sts)
    d_ok = c[("q", "OK")] - c[("f", "OK")]
    if c["changed"]: rows.append((op, n, c[("f","OK")], c[("q","OK")], c[("f","MISMATCH")], c[("q","MISMATCH")],
                                  c[("f","SKIP")], c[("q","SKIP")], c[("f","CRASH")], c[("q","CRASH")], d_ok))
for r in sorted(rows, key=lambda r: r[-1])[:35]:
    print(f"  {r[0]:32s} n={r[1]:4d}  OK {r[2]:4d}->{r[3]:<4d} MISM {r[4]:3d}->{r[5]:<3d} "
          f"SKIP {r[6]:3d}->{r[7]:<3d} CRASH {r[8]:2d}->{r[9]:<2d}  ΔOK={r[10]:+d}")

# new failure reasons introduced by quantization
newr = collections.Counter()
for j in changed:
    if qv[j] != "OK" and fv[j] == "OK":
        newr[(qv[j], QR[j][:70])] += 1
print("\ntop NEW failure reasons under --quantize (float was OK):")
for (s, r), n in newr.most_common(15):
    print(f"  {n:5d}  {s:9s} {r}")
json.dump({"paired": len(univ), "changed": len(changed),
           "float": {s: tot['f:'+s] for s in sts}, "int8": {s: tot['q:'+s] for s in sts}},
          open(PREF + "_summary.json", "w"), indent=1)
