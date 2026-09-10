#!/usr/bin/env bash
# QNN's AOT path hard-crashes the host process (known non-deterministic heap corruption),
# so drive the resumable sweep in a restart loop until it stops making progress.
cd /data/jwen929/mobile
source android-dev/android-env.sh >/dev/null 2>&1
export PATH="/data/jwen929/mobile/.venv/bin:$PATH" PYTHONPATH=/data/jwen929
prev=0
for i in $(seq 1 40); do
    .venv/bin/python tmp/quantab/ann_rate.py qualcomm corpus_v7/qnn_float tmp/quantab/ann_qnn.tsv 3 \
        >> tmp/quantab/ann_qnn.log 2>&1
    n=$(wc -l < tmp/quantab/ann_qnn.tsv)
    echo "pass $i: $n rows"
    [ "$n" -ge 215 ] && break
    if [ "$n" -le "$prev" ]; then
        # no progress: the crashing op would loop forever — record it and skip it
        awk -F'\t' 'NR>1{print $1}' tmp/quantab/ann_qnn.tsv > /tmp/.qnn_done_$$
        echo "STUCK after $n rows (crashing op not recorded); appending a PROCCRASH row"
        .venv/bin/python - "$n" <<'PY'
import re, sys, pathlib, collections
RE_G=re.compile(r"Graph: (.*)"); RE_OP=re.compile(r"n\d+\*?=([A-Za-z_][\w.]*)\(")
done={l.split("\t")[0] for l in open("tmp/quantab/ann_qnn.tsv").read().splitlines()[1:]}
ops=set()
for p in pathlib.Path("corpus_v7/qnn_float").glob("*/*.py"):
    g=RE_G.search(p.read_text(errors="replace")[:700])
    if not g: continue
    o=RE_OP.findall(g.group(1))
    if len(o)==1: ops.add(o[0])
nxt=sorted(o for o in ops if o not in done)
if nxt:
    open("tmp/quantab/ann_qnn.tsv","a").write(f"{nxt[0]}\t0\t0\t0\tPROCCRASH\n")
    print("skipped", nxt[0])
PY
    fi
    prev=$n
done
echo "done: $(wc -l < tmp/quantab/ann_qnn.tsv) rows"
