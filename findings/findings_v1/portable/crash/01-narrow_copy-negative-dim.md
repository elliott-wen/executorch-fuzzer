# Crash reason: narrow_copy / select with negative or out-of-range `dim`

**Signal:** SIGABRT (exit 134). **Count:** 4,043 by abort sub-pattern (`range [0,N] got -1`);
4,257 by op-presence (`narrow_copy` is the highest-priority dim-op in the graph).
**Largest single crash reason — ~71% of all crashes.**

## Abort signature (captured by the localizer)
```
[tensor_impl.h:136] In function size(), assert failed (dim < dim_ && dim >= 0):
Dimension out of range (expected to be in range of [0, N], but got -1)
```

## Mechanism
The portable `narrow_copy` (and `select`) argument validator calls `in.size(dim)` with the
**raw, un-normalized `dim`** (overwhelmingly `-1`). `TensorImpl::size()` is a fatal
`ET_CHECK_MSG` that aborts when `dim < 0`, so the intended graceful
`InvalidArgument`/`ET_KERNEL_CHECK` path is never reached. Eager PyTorch accepts negative
`dim` (normalizes it) or raises a catchable `IndexError`; portable hard-aborts.

Full kernel root cause (with `op_narrow_copy.cpp` / `slice_util.cpp` line citations):
see [../crash-narrow_copy.md](../crash-narrow_copy.md).

## Reproduce
```bash
.venv/bin/python tmp/run_portable/repro_one.py corpus/portable/w0/w0_1693.job; echo "exit=$?"
# -> [tensor_impl.h:136] ... got -1   (exit 134, SIGABRT)
```
Representative jobs: `w0:1693 w0:1697 w0:1159 w0:1334 w0:1395 w0:1391 w0:102 w0:1442`.

## Fix
Normalize and range-check `dim` (`dim = dim < 0 ? dim + in.dim() : dim;` then
`ET_KERNEL_CHECK(dim >= 0 && dim < in.dim())`) **before** any `in.size(dim)` call, so an
invalid dim returns a graceful error instead of aborting.

## Replay
Self-contained crash replay (runs the job in a child subprocess and asserts a fatal signal):
```bash
.venv/bin/python findings/portable/crash/replay_01-narrow_copy-negative-dim.py
```
Verified output:
```
job corpus/portable/w0/w0_1693.job  exit=134  signal=SIGABRT
  last stderr: [tensor_impl.h:136] In function size(), assert failed (dim < dim_ && dim >= 0): Dimension out of range (expected to be in range of [0, 0], but got -1
REPLAY: crash REPRODUCED (exit 134, matches documented)
```
Script: [replay_01-narrow_copy-negative-dim.py](replay_01-narrow_copy-negative-dim.py). Exit 0 iff the crash reproduces (any fatal signal).

## Classification
Real ExecuTorch portable robustness bug (missing guard → native abort where eager errors
cleanly). Amplified by the fuzzer emitting negative `dim` constants, but the abort itself is
a genuine kernel defect.
