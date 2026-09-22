# Crash reason: integer divide / modulo by zero → SIGFPE (NEW)

**Signal:** SIGFPE (exit 136 = 128+8). **Count:** 205 (3.4% of all crashes). **A distinct
crash class from the SIGABRT asserts — found by the crash localizer.**

## Why there is no abort message
Unlike the `tensor_impl.h` aborts, a SIGFPE is a **hardware trap** raised by the CPU on an
integer division by zero (or `INT_MIN / -1`). There is no ET_CHECK and no stderr message —
the process just dies with "Floating point exception (core dumped)". The localizer captured
this as an empty signature with signal 8, which is why these 205 were invisible to a
message-based triage.

## Mechanism
The portable integer `div` / `remainder` / `fmod` / `floor_divide` kernels perform a raw C++
integer `/` or `%`. When a divisor **element is 0** (in an integer tensor), the hardware
traps. Eager PyTorch raises a catchable error / handles it (and float div-by-zero gives
`inf`/`nan`, not a trap). Op presence across the 205 SIGFPE graphs:

| op present | graphs |
|---|---:|
| `remainder` | 75 |
| `div` | 53 |
| `fmod` | 49 |
| `floor_divide` | 40 |

(Several div/mod ops co-occur; the trap is the first integer-zero divisor reached.)

## Reproduce
```bash
.venv/bin/python tmp/run_portable/repro_one.py corpus/portable/w0/w0_626.job; echo "exit=$?"
# -> Floating point exception (core dumped)   exit=136
```
`w0:626` graph: `n11 = floor_divide.out(n5, L1, out=n2)` with an integer leaf `L1`
containing a 0. More SIGFPE jobs are in `tmp/run_portable/crashes.jsonl` (filter
`verdict == "SIGFPE"`).

## Distinction from the MISMATCH `floor_divide` finding
The mismatch deep-dive ([../bugs/floor_divide.md](../bugs/floor_divide.md)) covers the
*float* `0.0/0.0 → ±inf vs nan` value bug. This is the separate **integer** case: integer
`/0` does not produce a value at all — it **crashes** the process.

## Fix
Guard integer divisors for zero (and the `INT_MIN / -1` overflow case) in the portable
`div`/`remainder`/`fmod`/`floor_divide` integer paths — return the ATen-defined result or a
graceful `InvalidArgument` instead of executing a trapping hardware divide.

## Replay
Self-contained crash replay (runs the job in a child subprocess and asserts a fatal signal):
```bash
.venv/bin/python findings/portable/crash/replay_03-integer-divide-by-zero-sigfpe.py
```
Verified output:
```
job corpus/portable/w0/w0_626.job  exit=136  signal=SIGFPE
REPLAY: crash REPRODUCED (exit 136, matches documented)
```
(No stderr message — SIGFPE is a hardware trap.) Script:
[replay_03-integer-divide-by-zero-sigfpe.py](replay_03-integer-divide-by-zero-sigfpe.py).
Exit 0 iff the crash reproduces (any fatal signal).

## Classification
Real ExecuTorch portable robustness bug — a hard process crash on a data-dependent integer
zero divisor, where eager raises a recoverable error. Severity: high (uncatchable crash;
on-device this would take the whole inference process down).
