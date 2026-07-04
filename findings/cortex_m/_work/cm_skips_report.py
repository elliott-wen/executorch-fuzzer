#!/usr/bin/env python3
"""SKIP + CRASH reason aggregation per operator for cortex-m (analysis_single.md step 4/6).
The reason string IS the finding. Normalizes volatile bits so distinct reason SIGNATURES
group. Splits CRASH into INFRA (fvp_runner rc=5 build-fail / executor unavailable — NOT a
device bug) vs DEVICE (native abort / runtime error). Also emits a machine-readable summary.
"""
import re, collections, json
from pathlib import Path

W = Path("/data/jwen929/mobile/findings/cortex_m/_work")
man = {}
for ln in (W / "manifest.tsv").read_text().splitlines()[1:]:
    f = ln.split("\t")
    if len(f) >= 2:
        man[f[0]] = f[1]

def sig(reason):
    r = reason
    r = re.sub(r"/tmp/\S+", "/tmp/…", r)
    r = re.sub(r"/data/\S+", "/data/…", r)
    r = re.sub(r"ops=[0-9a-f]{4,}", "ops=…", r)
    r = re.sub(r"0x[0-9a-fp]+", "0x…", r)
    r = re.sub(r"\b\d+\b", "N", r)
    return r.strip()[:220]

INFRA_RE = re.compile(r"rc=5|executor unavailable|build failed|could not probe|FATAL: no FVP")

skip_by = collections.defaultdict(collections.Counter)
crash_dev = collections.defaultdict(collections.Counter)
crash_infra = collections.defaultdict(collections.Counter)
tally = collections.Counter()
for ln in (W / "skip.tsv").read_text().splitlines()[1:]:
    f = ln.split("\t")
    if len(f) < 2:
        continue
    st, jid = f[0], f[1]
    reason = f[2] if len(f) > 2 else ""
    op = man.get(jid, "?")
    s = sig(reason)
    if st == "SKIP":
        skip_by[op][s] += 1; tally["SKIP"] += 1
    elif st == "CRASH":
        if INFRA_RE.search(reason):
            crash_infra[op][s] += 1; tally["CRASH_INFRA"] += 1
        else:
            crash_dev[op][s] += 1; tally["CRASH_DEVICE"] += 1
    elif st == "TIMEOUT":
        crash_dev[op][sig("TIMEOUT " + reason)] += 1; tally["TIMEOUT"] += 1

def dump(title, tbl):
    print(f"\n{'='*90}\n{title}\n{'='*90}")
    for op in sorted(tbl, key=lambda o: -sum(tbl[o].values())):
        for s, c in tbl[op].most_common():
            print(f"  {c:>5}  {op:<32} {s}")

print("TALLY:", dict(tally))
dump("SKIP reasons (per op, per reason signature)", skip_by)
dump("CRASH — DEVICE (native abort / runtime error — the missing guard IS the bug)", crash_dev)
dump("CRASH — INFRA (fvp_runner rc=5 / executor unavailable — NOT device bugs; re-run)", crash_infra)

summary = {
    "tally": dict(tally),
    "skip_ops": {op: dict(c) for op, c in skip_by.items()},
    "crash_device_ops": {op: dict(c) for op, c in crash_dev.items()},
    "crash_infra_ops": {op: dict(c) for op, c in crash_infra.items()},
}
(W / "skips_summary.json").write_text(json.dumps(summary, indent=1))
print(f"\nwrote {W/'skips_summary.json'}")
