# Crash reason: unhandled ComplexFloat into generic elementwise (`dtype_util.h`)

**Signal:** **SIGSEGV (exit 139 = 128+11)** — VERIFIED by replay. **Count:** 1–2 (from the
long tail). Split out of the former
[04 long-tail file](04-tail-dtype-and-memory-corruption.md).

## Abort signature (captured by the localizer)
```
[dtype_util.h:37] Unhandled dtype ComplexFloat for generic_elementwise_op
```

## Mechanism
`_fft_r2c` produces a **complex** (`ComplexFloat`) tensor, which a downstream elementwise op
feeds into the generic elementwise dtype dispatch. That dispatch has no `ComplexFloat` case;
it logs the unhandled-dtype line and then dereferences an unpopulated path, dying with
SIGSEGV. Example job `w17:547` chains `_fft_r2c`, `cumsum`, `remainder`.

## Replay
Self-contained crash replay (runs the job in a child subprocess and asserts a fatal signal):
```bash
.venv/bin/python findings/portable/crash/replay_07-complexfloat-elementwise.py
```
Verified output:
```
job corpus/portable/w17/w17_547.job  exit=139  signal=SIGSEGV
  last stderr: [dtype_util.h:37] Unhandled dtype ComplexFloat for generic_elementwise_op
REPLAY: crash REPRODUCED (exit 139, matches documented)
```
Script: [replay_07-complexfloat-elementwise.py](replay_07-complexfloat-elementwise.py).
Exit 0 iff the crash reproduces (any fatal signal).

## Fix
Reject complex inputs gracefully at the elementwise dtype dispatch (return `InvalidArgument`
for `ComplexFloat`/`ComplexDouble`) instead of falling through to an unhandled path that
segfaults.

## Classification
Real ExecuTorch portable robustness bug — an unhandled dtype that segfaults rather than
erroring cleanly. Severity: high (memory-unsafe fall-through, not a clean assert).
