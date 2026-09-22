# Crash reason: scalar_type mismatch fatal Check (`tensor_util.h`)

**Signal:** SIGABRT (exit 134 = 128+6). **Count:** ~13 (from the long tail). Split out of the
former [04 long-tail file](04-tail-dtype-and-memory-corruption.md).

## Abort signature (captured by the localizer)
```
[tensor_util.h:595] Check failed (a.scalar_type() == b.scalar_type()):
Tensors do not match: dtype={Half, Long}   (also {Float, Long}, {Bool, Long}, {Half, Float})
```

## Mechanism
An op that requires its two tensor operands to share a dtype hit a fatal `ET_CHECK` (not a
graceful `ET_KERNEL_CHECK`) in `tensor_util.h` on mismatched dtypes, and aborted. Eager
PyTorch type-promotes the operands instead of aborting, so the same graph succeeds in eager.

## Replay
Self-contained crash replay (runs the job in a child subprocess and asserts a fatal signal):
```bash
.venv/bin/python findings/portable/crash/replay_05-scalar-type-mismatch.py
```
Verified output:
```
job corpus/portable/w16/w16_1264.job  exit=134  signal=SIGABRT
  last stderr: [tensor_util.h:595] Check failed (a.scalar_type() == b.scalar_type()): Tensors do not match: dtype={Half, Long}
REPLAY: crash REPRODUCED (exit 134, matches documented)
```
Script: [replay_05-scalar-type-mismatch.py](replay_05-scalar-type-mismatch.py). Other example
jobs: `w25:721`, `w29:159`. Exit 0 iff the crash reproduces (any fatal signal).

## Fix
Convert this internal invariant `ET_CHECK` into a graceful dtype guard (`ET_KERNEL_CHECK` →
`InvalidArgument`), or type-promote the operands to match eager semantics.

## Classification
Real ExecuTorch portable robustness bug — a fatal abort on a mismatched-dtype operand pair
where eager type-promotes. Severity: medium (deterministic abort, no memory unsafety).
