# ARM-only crash: `_log_softmax`/`add.Scalar` version gate → shared `narrow_copy`/`unfold` crash

**Class:** version confound (NOT a NEON bug; NOT a new crash bug) · **Verdict on phone:** CRASH ·
**Scope:** 109 of the 111 ARM-only crashes (80× `_log_softmax`, 26× `add.Scalar`, …)

## Buggy model graph (run to trigger on the phone)
```bash
.venv/bin/python findings/xnnpack-arm/bugs/graph_arm-only-logsoftmax-version-crash.py
# or:
python feed.py w0:480 --corpus corpus/xnnpack --host 127.0.0.1
```
**Expected:** `status CRASH  "executor process died (native abort)"` on the phone, while the same
job is `SKIP` on the x86 host. (Prereq: broker + ARM phone connected.)

## What happens (and why it is NOT what it first looked like)
These jobs **SKIP on x86 but CRASH on ARM**. The graphs contain the shared crash-prone ops
(`w0:480` → `_log_softmax → narrow_copy×3 → floor_divide`; `w0:1095` → `unfold_copy`). The
difference is **which ExecuTorch build runs the `.pte`**, not the architecture:

- **x86 host** = pip wheel `executorch 1.4.0.dev20260625`, whose `op_log_softmax.cpp:152` has a
  **`Check failed (false)` hard-disable**. It fires at instruction 0:0 → the whole
  `method->execute()` returns error `0x12` → **SKIP**, before the crash-prone ops ever run.
- **ARM phone** = our source-built AAR from `pytorch_ref/executorch @ 2759ef1`, whose
  `op_log_softmax.cpp` **fully implements** the op (ET_KERNEL_CHECKs only, no line-152 disable).
  So `_log_softmax` runs, execution continues, and it reaches the **shared `narrow_copy` /
  `unfold_copy` native-abort** ([../../portable/crash/01-narrow_copy-negative-dim.md](../../portable/crash/01-narrow_copy-negative-dim.md),
  [../../portable/crash/02-unfold_copy-zero-dim.md](../../portable/crash/02-unfold_copy-zero-dim.md)).

So: the **crash is the documented shared bug**; the ARM-vs-x86 difference is the **`_log_softmax`
portable-kernel version**. (My first guess — a graceful-check-turned-fatal on device — was wrong;
the source diff disproved it.)

## The broader caveat this exposed
The whole ARM-vs-x86 comparison has a **version confound**: x86 baseline = pip wheel, ARM = source
AAR `@2759ef1`. Portable-kernel verdict diffs (like this one) can be version, not arch. The
XNNPACK **delegate microkernel** findings ([vlog](neon-vlog-special-values.md),
[gelu](neon-gelu-nan-launder.md), [tan/erf](neon-tan-erf-precision.md)) are still genuinely arch
(arch-selected kernels, source-verified). See the caveat box in
[../00-distribution.md](../00-distribution.md).

## Fix / next step
Not an ARM bug to fix per se. To remove the confound and get a clean arch-only comparison,
**rebuild the x86 baseline from the same `@2759ef1` source** (install `pytorch_ref/executorch`
into the venv) and re-run the x86 corpus. The underlying `narrow_copy`/`unfold_copy` aborts are
already tracked in the portable crash findings.
