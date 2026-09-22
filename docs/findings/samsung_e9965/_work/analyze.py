#!/usr/bin/env python3
"""analysis_single.md steps 1-3 for the samsung/E9965 (Exynos 2600 ENN) run.

Reads a feed skip-log (non-OK rows only) + the corpus delegation headers, buckets every
non-OK outcome by operator, and splits delegated (ops>=1 = ENN kernel ran) from portable
(ops=0 = CPU fallback, NOT an ENN bug). Emits the per-operator table and triage worklists.

Usage: analyze.py <skiplog.tsv> <corpus_dir>
"""
import sys, os, re, json, collections

skiplog, corpus = sys.argv[1], sys.argv[2]

def job_py(job_id):
    # "w1:9" -> corpus/w1/w1_9.py
    shard, n = job_id.split(":")
    return os.path.join(corpus, shard, f"{shard}_{n}.py")

_deleg_cache = {}
def delegated_ops(job_id):
    """Return #ops the ENN delegate absorbed for this job (-1 if header unreadable)."""
    if job_id in _deleg_cache:
        return _deleg_cache[job_id]
    ops = -1
    try:
        with open(job_py(job_id)) as f:
            for line in f:
                m = re.search(r'# delegated: backend=\w+ ops=(\d+)', line)
                if m:
                    ops = int(m.group(1)); break
                if not line.startswith('#') and line.strip():
                    break
    except FileNotFoundError:
        pass
    _deleg_cache[job_id] = ops
    return ops

def op_of(op_chain):
    # "n0*=add.Scalar(L0,·,·)" -> "add.Scalar"
    m = re.search(r'=([A-Za-z0-9_.]+)\(', op_chain)
    return m.group(1) if m else op_chain

def skip_class(reason):
    if 'Operator missing' in reason or '0x14' in reason:
        return 'SKIP:missing-operator'
    if 'Execution failed for method' in reason or '0x1]' in reason:
        return 'SKIP:exec-failed'
    if 'feed compare' in reason or 'non-tensor' in reason:
        return 'SKIP:feeder-side'
    return 'SKIP:other'

def mism_class(reason):
    if 'non-finite' in reason:
        return 'MISMATCH:nonfinite'
    return 'MISMATCH:finite'

# op -> {status_class: count}, and delegated/portable tallies per op
tbl = collections.defaultdict(lambda: collections.Counter())
deleg = collections.defaultdict(lambda: collections.Counter())  # op -> {'deleg':n,'port':n,'unk':n}
rows_by_op = collections.defaultdict(list)  # op -> [(status, job, ops, reason)]

n = 0
with open(skiplog) as f:
    header = f.readline()
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 4:
            continue
        status, job, reason, op_chain = parts[0], parts[1], parts[2], parts[3]
        op = op_of(op_chain)
        ops = delegated_ops(job)
        bucket = ('deleg' if ops >= 1 else 'port' if ops == 0 else 'unk')
        deleg[op][bucket] += 1
        if status == 'MISMATCH':
            cls = mism_class(reason)
        elif status == 'SKIP':
            cls = skip_class(reason)
        elif status == 'CRASH':
            cls = 'CRASH'
        else:
            cls = status
        tbl[op][cls] += 1
        if cls != status:
            tbl[op][status] += 1
        rows_by_op[op].append((status, job, ops, reason))
        n += 1

print(f"# non-OK rows: {n}   distinct ops: {len(tbl)}\n")

# Overall status split
tot = collections.Counter()
for op, c in tbl.items():
    for k in ('MISMATCH','CRASH','SKIP'):
        tot[k] += c[k]
print("== overall non-OK ==", dict(tot))

# Delegated vs portable for MISMATCH/CRASH (the 3a filter)
print("\n== 3a: delegated(ENN) vs portable(CPU fallback) among MISMATCH+CRASH ==")
dm=collections.Counter()
for op, rows in rows_by_op.items():
    for status, job, ops, reason in rows:
        if status in ('MISMATCH','CRASH'):
            dm['deleg' if ops>=1 else 'port' if ops==0 else 'unk'] += 1
print(dict(dm))

# Per-op table, ENN-delegated failures first (the real findings)
def deleg_fail(op):
    return sum(1 for s,j,o,r in rows_by_op[op] if o>=1 and s in ('MISMATCH','CRASH'))

print("\n== per-op: ENN-DELEGATED failures (ops>=1), ranked ==")
print(f"{'operator':32} {'MIS-fin':7} {'MIS-nf':6} {'CRASH':5} {'SKIP':4}  deleg/port")
for op in sorted(tbl, key=lambda o:(-deleg_fail(o), o)):
    c = tbl[op]
    df = deleg_fail(op)
    if df == 0:
        continue
    finmis = sum(1 for s,j,o,r in rows_by_op[op] if o>=1 and s=='MISMATCH' and 'non-finite' not in r)
    nfmis  = sum(1 for s,j,o,r in rows_by_op[op] if o>=1 and s=='MISMATCH' and 'non-finite' in r)
    crash  = sum(1 for s,j,o,r in rows_by_op[op] if o>=1 and s=='CRASH')
    skip   = sum(1 for s,j,o,r in rows_by_op[op] if o>=1 and s=='SKIP')
    print(f"{op:32} {finmis:7} {nfmis:6} {crash:5} {skip:4}  {deleg[op]['deleg']}/{deleg[op]['port']}")

print("\n== SKIP reasons (delegated), grouped ==")
skipr = collections.Counter()
for op, rows in rows_by_op.items():
    for s,j,o,r in rows:
        if s=='SKIP' and o>=1:
            skipr[skip_class(r)] += 1
print(dict(skipr))
