# Crash reason: memory corruption (the most severe class)

**Signal:** SIGABRT (glibc heap abort, 134) or SIGSEGV (139). **Count:** ≈5 (from the long
tail) — small but the **highest-priority** crashes. Split out of the former
[04 long-tail file](04-tail-dtype-and-memory-corruption.md).

## Mechanism
These are **out-of-bounds writes / heap corruption**, not asserts — a kernel scribbled past a
buffer before any check fired. Two surface as glibc abort messages (`double free or
corruption`, `free(): invalid next size`) once the corrupted heap metadata is next touched;
two surface as a raw SIGSEGV with no message. Likely candidates are an out-of-bounds write in
a copy/pad/scatter kernel fed out-of-range indices or sizes (cf. the
`max_pool2d_with_indices_backward` OOB in [../bugs/maxpool2d-backward.md](../bugs/maxpool2d-backward.md)).

| job | signal | signature | ops in chain |
|---|---|---|---|
| `w56:557` | SIGABRT | `double free or corruption (out)` | narrow_copy, constant_pad_nd, acosh, log |
| `w92:1487` | SIGABRT | `double free or corruption (out)` / `free(): invalid next size` | floor_divide, max, fmod, sum, log10 |
| `w10:359` | SIGSEGV | (no message) | cumsum, floor, slice, clampr |
| `w38:1077` | SIGSEGV | (no message) | fmod, remainder, sin, slice_copy, argmax |

## Replay
Self-contained crash replay — iterates over **all four** jobs and asserts each crashes:
```bash
.venv/bin/python findings/portable/crash/replay_08-memory-corruption.py
```
Verified output:
```
job corpus/portable/w56/w56_557.job  exit=134  signal=SIGABRT  (documented 134)
  last stderr: double free or corruption (out)
job corpus/portable/w92/w92_1487.job  exit=134  signal=SIGABRT  (documented 134)
  last stderr: double free or corruption (out)
job corpus/portable/w10/w10_359.job  exit=139  signal=SIGSEGV  (documented 139)
job corpus/portable/w38/w38_1077.job  exit=139  signal=SIGSEGV  (documented 139)
REPLAY: 4/4 jobs crashed (fatal signal)
REPLAY: ALL memory-corruption jobs REPRODUCED
```
Script: [replay_08-memory-corruption.py](replay_08-memory-corruption.py). Exit 0 iff **all**
jobs reproduce (any fatal signal each).

## Fix
Reproduce each under ASan to pin the offending kernel and bounds check, then add the missing
bounds guard on the OOB index/size in the copy/pad/scatter kernel.

## Classification
Real ExecuTorch portable memory-safety bug — these are out-of-bounds writes, worse than an
assert because they can silently corrupt state (or be exploitable) rather than failing fast.
Severity: critical.
