#!/usr/bin/env python3
"""Per-operator table (analysis_single.md step 2) for cortex-m.

Joins skip.tsv (feed outcomes) with manifest.tsv (op per job) into a per-operator ledger:
one row per operator with OK / MISMATCH / CRASH / SKIP / TIMEOUT counts over its k samples.
cortex-m has no delegated/portable split at the header (pass-based), so instead of a
delegation column we tag the per-op cortex_m COMPUTE-capability statically (the cm_kernels
re-lowering gives the exact per-job signal for the non-OK worklist separately).

Also splits CRASH into infra (fvp_runner rc=5 build fail / executor unavailable / TIMEOUT-ish)
vs device native abort, per the rigor reminder about not counting infra dropouts as crashes.
"""
import collections
from pathlib import Path

W = Path("/data/jwen929/mobile/findings/cortex_m/_work")
man = {}
op_total = collections.Counter()
for ln in (W / "manifest.tsv").read_text().splitlines()[1:]:
    f = ln.split("\t")
    if len(f) >= 2:
        man[f[0]] = f[1]
        op_total[f[1]] += 1

per_op = collections.defaultdict(lambda: collections.Counter())
infra_crash = collections.Counter()   # op -> count of rc=5 / executor-unavailable
nonok_jobs = collections.defaultdict(list)   # (op,status) -> [job_id...]
skip = W / "skip.tsv"
rows = skip.read_text().splitlines()[1:] if skip.exists() else []
for ln in rows:
    f = ln.split("\t")
    if len(f) < 2:
        continue
    st, jid = f[0], f[1]
    reason = f[2] if len(f) > 2 else ""
    op = man.get(jid, "?")
    per_op[op][st] += 1
    nonok_jobs[(op, st)].append(jid)
    if st == "CRASH" and ("rc=5" in reason or "executor unavailable" in reason or "build failed" in reason):
        infra_crash[op] += 1

# emit the table sorted by (MISMATCH+CRASH+SKIP) desc
def nonok(op):
    c = per_op[op]
    return c["MISMATCH"] + c["CRASH"] + c["SKIP"] + c["TIMEOUT"]

ops = sorted(per_op, key=lambda o: (-nonok(o), o))
print(f"{'operator':<34}{'k':>6}{'OK':>7}{'MISM':>6}{'CRASH':>6}{'(infra)':>8}{'SKIP':>6}{'TO':>4}")
tot = collections.Counter()
for op in ops:
    c = per_op[op]
    k = op_total.get(op, 0)
    ok = k - nonok(op)
    print(f"{op:<34}{k:>6}{ok:>7}{c['MISMATCH']:>6}{c['CRASH']:>6}{infra_crash[op]:>8}{c['SKIP']:>6}{c['TIMEOUT']:>4}")
    for kk in ("MISMATCH", "CRASH", "SKIP", "TIMEOUT"):
        tot[kk] += c[kk]
tot["infra"] = sum(infra_crash.values())
print(f"\nTOTAL non-OK: MISMATCH={tot['MISMATCH']} CRASH={tot['CRASH']} (infra rc5/unavail={tot['infra']}) "
      f"SKIP={tot['SKIP']} TIMEOUT={tot['TIMEOUT']}")
print(f"operators with >=1 non-OK: {len(ops)}")

# dump the non-OK job lists for downstream classification
import json
(W / "nonok_jobs.json").write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in nonok_jobs.items()}))
print(f"wrote {W/'nonok_jobs.json'}")
