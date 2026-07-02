# Why the corpus run didn't surface the copy-elision aliasing bug

The [copy-elision / memory-planning aliasing bug](bugs/vulkan-copy-elision-aliasing.md) is real and
device-confirmed. Yet it did not pop out of the 100k-graph differential-fuzz run as a clean signal.
Here is why — verified, not guessed (`tmp/run_vulkan_galaxy_26/why_missed.py`).

## It was almost certainly *hit*, not missed

The bug's structural shape — one tensor fanned out to ≥2 returned outputs, one of which is a copy
op — is **common** in the corpus:

```
graphs scanned: 4000 (sample)
  node fans out to >=2 returned outputs:            3617  (90.4%)
  ...one of those outputs is a copy op (bug shape):  247  (6.2%)   -> ~6,200 graphs corpus-wide
```

So ~6% of the corpus carries the exact pattern. Their corrupted outputs are sitting inside the
**8,202 MISMATCH** pile right now. The corpus caught the *symptom*. What it couldn't do is *name the
cause*. Three reasons:

## 1. The localizer's model can't attribute a planner bug

The first-divergence localizer assumes **"the first node whose value differs from eager = the buggy
op."** That holds for kernel bugs. A memory-aliasing bug breaks the assumption: the wrong value
appears at an *innocent* op (`acos`/`prod`) whose own kernel is correct and whose inputs *looked*
correct when produced. There is no "born-here op" — the culprit is the planner, which is not a graph
node. So the localizer mislabels each instance as a numeric `delta` on some ordinary op and bins it
with fp16/int64 noise. The bug has no distinct fingerprint in the per-op root table.

## 2. No structural A/B mutation in the pipeline

What actually isolates the bug is the **copy ↔ fresh-buffer swap** (`Bug` vs `Control`): identical
math, different buffer lifetimes, different result. Differential-vs-eager fuzzing only ever compares
*one* lowering against eager — it has no mutation that holds the math fixed and perturbs the
*memory plan*. Without that A/B, a corrupted `0.8486 → 2.4668` is indistinguishable from any other
wrong number.

## 3. The bug is fragile to instrumentation

Corruption depends on the exact planning state. In `why_missed.py`, returning a different set of
intermediates flips it on/off (one probe accidentally dead-code-eliminated the copy's consumer and
the corruption vanished). So even when the localizer re-lowers a graph to inspect it, the act of
changing the output set can move or hide the corruption — a per-node value probe is the wrong
instrument for a whole-graph planning bug.

## Contributing factor: some instances crash instead of mismatching

`clone`/`alias_copy`/`lift_fresh_copy` are also over-represented in **CRASH** graphs (~1.7× baseline,
[00-distribution.md](00-distribution.md)). Buffer reuse that lands on an out-of-bounds region aborts
rather than returning a wrong value, scattering the signal across both the MISMATCH and CRASH piles.

## What would catch it next time

- A **memory-plan mutation**: for each graph, lower it twice — once normally, once with copy-elision
  / buffer-reuse disabled (or copies replaced by fresh-buffer ops) — and diff the two *device* runs.
  A divergence with identical math = a planning bug. (This is the corpus-scale version of `Bug` vs
  `Control`.)
- An **alias-shadowing oracle**: tag each elided copy's source buffer and assert it isn't written
  while still live.
- Treat **silent wrong-data in all-correct-kernel graphs** as its own bucket rather than folding it
  into per-op numeric clusters.
