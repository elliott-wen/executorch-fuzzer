# SHAPE verdicts — investigated and RULED OUT

The `corpus_v4/vgf` graph-opt analysis recorded **13 rows with mechanism `SHAPE`** (12 usable:
all `COMPOSITIONAL` + `AB_OK`; the 1 `SIBLING_DEPENDENT` row is `A_ALSO_DIVERGES`, i.e. an
operator bug, already excluded). `GRAPHOPT_REPORT.md` §4/§5 called these "a distinct, serious
graph-opt defect worth its own follow-up".

**They are not a shape defect.** Four independent checks, all negative.

## Method and results

Env: `MODEL_CONVERTER_LIB_DIR=/opt/nvidia/nsight-systems/2024.6.2/host-linux-x64`,
`PATH=$MOBILE/.venv/bin:$PATH` (the model-converter needs GLIBCXX_3.4.30; without it the vgf
backend probes as unavailable and every `build_job` returns `SKIP: unknown backend 'vgf'`).

### 1. `shape_probe.py` — AOT plan vs eager, ORIGINAL jobs
Lowers each of the 12 corpus jobs and compares the `.pte` execution plan's **declared** output
sizes against the eager reference. **12/12 `PLAN==EAGER`.** Also: `user_pos` is the identity list
`[0..n-1]` and `n_plan_out == n_eager` in every case, so there are no prepended mutated-input
aliases and no positional misalignment at the plan level.

### 2. `shape_probe2.py` — AOT plan vs eager, RECONSTRUCTED A/B graphs
The verdicts came from `cone_localizer` sub-graphs, not the original jobs, so both A (`build_src([])`)
and B (`build_src(needed_live)`) were lowered too — 24 graphs. **24/24 `PLAN==EAGER`.**
Note `build_src` returns `return (target,)`, a single output, so `analyze_comp`'s `[-1]` indexing
is safe. Only `w22:81` B-side has 2 plan outputs (`user_pos=[1]`), and both have equal numel.

### 3. `shape_run.py` — device output byte count
Runs each B-side `.pte` on the VGF runtime (`vgf_client/vgf_runner.sh`, lavapipe + ML SDK
emulation layer) and compares the emitted `out_*.bin` size against `eager.numel() *
element_size()`. **12/12 byte counts match exactly.** The device never returns a wrongly-sized
tensor.

### 4. `shape_final.py` — full comparison replay, N=3
Parses the USER_OUTPUT back into a tensor and runs the real comparison. **0/12 are SHAPE.**

| job | target | verdict now (3 reps) | reassign to |
|---|---|---|---|
| w103:715 | `unfold_copy` | OK, OK, OK | *drop (flaky)* |
| w104:305 | `neg` | OK, OK, OK | *drop (flaky)* |
| w10:591 | `div` | OK, OK, OK | *drop (flaky)* |
| w11:655 | `div` | OK, OK, OK | *drop (flaky)* |
| w106:621 | `clamp` | MISMATCH/VALUE ×3 | NumPlan |
| w107:319 | `log` | MISMATCH/VALUE ×3 | NumPlan |
| w114:256 | `min` | MISMATCH/VALUE ×3 | NumPlan |
| w120:300 | `squeeze_copy` | MISMATCH/VALUE ×3 | NumPlan |
| w121:253 | `reciprocal` | MISMATCH/VALUE ×3 | NumPlan |
| w22:81 | `view_copy` | MISMATCH/VALUE ×3 | NumPlan |
| w107:200 | `cumsum` | MISMATCH/ZEROED ×3 | Alias |
| w111:159 | `scatter_add` | MISMATCH/ZEROED ×3 | Alias |

**The four that now pass are exactly the four rows recorded with `n_div=1/3`** in
`graphopt_all.tsv` (w103:715, w104:305, w10:591, w11:655). The determinism gate had already
flagged them; they were carried into the category anyway.

## Where the spurious SHAPE label came from

`mech()` in `tmp/vgf_go_deep.py:58-60` assigns `SHAPE` on a bare element-count test:

```python
r = ref.flatten().float(); d = dev.flatten().float()
if r.numel() != d.numel(): return "SHAPE"
```

and `net/compare.py:select()` **silently falls back to returning every output** when `user_pos`
does not index into the tensor list it was handed:

```python
if user_pos is not None and all(0 <= i < len(et) for i in user_pos):
    return [et[i] for i in user_pos]
return et            # <-- silent fallback
```

If the worker ever returned a different number of tensors than `user_pos` expects, `select`
returns the wrong tensor, `mech` sees a different element count, and the case is labelled a
shape bug. This is a **comparison-path artifact**: the plan declares the right shape (checks 1–2)
and the device produces the right number of bytes (check 3). It is a sufficient mechanism for the
label, though the original run's client path could not be replayed exactly to confirm it fired.

## Recommended fixes
- `net/compare.py::select` — raise on an out-of-range `user_pos` instead of falling back; a silent
  substitution of the wrong output is worse than an error.
- `mech()` — separate "output shapes differ" from "compared the wrong output"; assert
  `len(ref) == len(dev)` before indexing.
- `GRAPHOPT_REPORT.md` §4/§5 — remove the SHAPE claim, or cite this directory.
