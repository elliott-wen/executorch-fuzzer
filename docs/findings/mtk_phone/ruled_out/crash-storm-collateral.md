# Ruled out: ~4,300 of the 4,517 bulk CRASHes are crash-storm collateral (3b)

The bulk feed (`--window 16`) logged **4,517 CRASH**. Re-running that exact set **serially**
(`--window 1`) returned **565 CRASH / 3,610 OK / 342 MISMATCH**. A 5×-per-job determinism gate on a
sample of the non-`copy` serial-crashers (linear, squeeze_copy, div, pow, bmm, sigmoid, group_norm,
unsqueeze_copy, reciprocal, mm, mean, tanh) returned **OK 5/5 for every one**.

**Mechanism:** when a job native-aborts the phone executor process, every in-flight neighbour is logged
CRASH; in serial mode the very next job(s) fed before the client restarts the executor are logged CRASH
too. `copy` (217 jobs, unconditional crasher) scattered through the run taints its neighbours.

**Verdict:** the only determinism-confirmed crash is **`copy`** (217/217, `../bugs/copy.md`). The
remaining ~4,300 CRASH rows are dropouts/restart artifacts, not crashes. (1 job was
`executor unavailable`; all others were genuine `native abort` — but only `copy`'s reproduce.)
Counting the raw column overstates the crash count **~21×**.
