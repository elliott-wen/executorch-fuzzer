# Crash reason: reduce_util.cpp path → SIGFPE (VERIFIED true signal)

**Signal:** **SIGFPE (exit 136 = 128+8)** — VERIFIED by replay, *not* SIGABRT. **Count:** ~10
(from the long tail). Split out of the former
[04 long-tail file](04-tail-dtype-and-memory-corruption.md).

## Signal correction
The crash localizer keyed these jobs on the line `[reduce_util.cpp:380] dtype is N` and
recorded them ambiguously. Running them shows that line is a **non-fatal** `ET_LOG` — the
process keeps going and then dies with **SIGFPE (136)**, an integer divide-by-zero hardware
trap, *not* a fatal dtype `ET_CHECK` abort. All three checked jobs (`w12:2081`, `w16:2`,
`w19:1158`) return -8 (SIGFPE).

## Mechanism
A reduction helper over an integer tensor computes a per-element divisor (e.g. a
mean/normalization count) that is **zero** and performs a raw integer `/`, which the CPU
traps. There is no stderr abort message beyond the informational `reduce_util.cpp` log line.
This is the same hardware-trap class as the integer divide-by-zero finding
([03-integer-divide-by-zero-sigfpe.md](03-integer-divide-by-zero-sigfpe.md)), reached via a
reduction path rather than an explicit `div`/`remainder` op.

## Replay
Self-contained crash replay (runs the job in a child subprocess and asserts a fatal signal):
```bash
.venv/bin/python findings/portable/crash/replay_06-reduce-util-unhandled-dtype.py
```
Verified output:
```
job corpus/portable/w12/w12_2081.job  exit=136  signal=SIGFPE
  last stderr: [reduce_util.cpp:380] dtype is 5
REPLAY: crash REPRODUCED (exit 136, matches documented)
```
Script: [replay_06-reduce-util-unhandled-dtype.py](replay_06-reduce-util-unhandled-dtype.py).
Other example jobs: `w16:2`, `w19:1158`. Exit 0 iff the crash reproduces (any fatal signal).

## Fix
Guard the reduction divisor for zero (and the `INT_MIN / -1` overflow case) before the
integer divide — return the ATen-defined result or a graceful `InvalidArgument` instead of
executing a trapping hardware divide. Same fix family as the integer div-by-zero finding.

## Classification
Real ExecuTorch portable robustness bug — an uncatchable hardware trap in a reduction helper
on a data-dependent zero divisor, where eager handles the case. Severity: high (uncatchable
crash; takes the inference process down).
