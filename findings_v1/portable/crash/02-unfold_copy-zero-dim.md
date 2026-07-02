# Crash reason: unfold_copy (and dim-ops) on a 0-D scalar input

**Signal:** SIGABRT (exit 134). **Count:** 1,740 by abort sub-pattern (`range [0,-1] got 0`);
1,527 by op-presence (`unfold_copy`). **~29% of all crashes.**

## Abort signature (captured by the localizer)
```
[tensor_impl.h:136] In function size(), assert failed (dim < dim_ && dim >= 0):
Dimension out of range (expected to be in range of [0, -1], but got 0)
```
The `[0, -1]` range is the tell-tale of a **0-D (scalar) tensor**: its valid dim range is
empty, so even `size(0)` is out of range.

## Mechanism
The portable `unfold_copy` validator (`check_unfold_copy_args`) accepts `dim == 0` for a 0-D
tensor (`tensor_has_dim` allows `d == 0` for 0-D), then evaluates `size <= self.size(dim)` →
`self.size(0)` on a 0-D tensor → fatal `ET_CHECK` in `TensorImpl::size()` → abort. The
triggering input is a reduction's 0-D scalar output (`min`/`prod`/`sum`/`argmax`…) fed into
`unfold_copy(x, 0, ...)`. Eager treats a 0-D input as 1-D and returns a `[0]`/`[1]`-shaped
tensor without error.

Full kernel root cause (with `copy_ops_util.cpp` line citations):
see [../crash-unfold_copy.md](../crash-unfold_copy.md).

## Reproduce
```bash
.venv/bin/python tmp/run_portable/repro_one.py corpus/portable/w0/w0_104.job; echo "exit=$?"
# -> [tensor_impl.h:136] ... range [0, -1], got 0   (exit 134, SIGABRT)
```
Representative jobs: `w0:104 w0:1095 w0:1701 w0:1193 w0:1302 w0:1418 w0:1534 w0:1684`.

## Fix
Add a 0-D (`nonzero_dim`-aware) guard before `self.size(dim)` — match eager's 1-D promotion,
or return a graceful `InvalidArgument`. Same class of fix as
[narrow_copy](01-narrow_copy-negative-dim.md): validate `dim` against the tensor rank before
calling `size()`.

## Replay
Self-contained crash replay (runs the job in a child subprocess and asserts a fatal signal):
```bash
.venv/bin/python findings/portable/crash/replay_02-unfold_copy-zero-dim.py
```
Verified output:
```
job corpus/portable/w0/w0_104.job  exit=134  signal=SIGABRT
  last stderr: [tensor_impl.h:136] In function size(), assert failed (dim < dim_ && dim >= 0): Dimension out of range (expected to be in range of [0, -1], but got 0
REPLAY: crash REPRODUCED (exit 134, matches documented)
```
Script: [replay_02-unfold_copy-zero-dim.py](replay_02-unfold_copy-zero-dim.py). Exit 0 iff the crash reproduces (any fatal signal).

## Classification
Real ExecuTorch portable robustness bug (missing 0-D guard → native abort where eager
succeeds). Amplified by the generator chaining reductions into `unfold_copy`.
