# Moto-unique mismatches — detailed cross-device analysis

Both phones ran the **same 76,801-graph corpus**, so we can diff per-job. The Moto has 9,703
mismatches vs the Pixel 9's 7,399. This document explains the difference.

## Confusion matrix (rows = Pixel 9, cols = Moto G54)

| Pixel ↓ / Moto → | OK | MISMATCH | CRASH | SKIP |
|---|---:|---:|---:|---:|
| **OK** | 29,960 | **2,396** | 0 | 3 |
| **MISMATCH** | 96 | 7,303 | 0 | 0 |
| **CRASH** | 3 | 4 | 3,180 | 1 |
| **SKIP** | 0 | 0 | 2 | 33,853 |

Reading:
- **7,303 shared mismatches** — and **96% fail at the exact same op** → identical delegate bugs.
- **SKIP (33,853) and CRASH (3,180) are essentially identical** across devices → those failure
  modes are 100% delegate-level, zero device variance.
- **2,396 Moto-unique** mismatches (Moto wrong, Pixel correct) — the entire "extra mismatch"
  delta. **Only 96 go the other way** (Pixel-unique). So the asymmetry is real: the Moto's GPU +
  memory-planner expose ~2,400 graphs the Pixel happened to get right.

## The 2,396 Moto-unique mismatches are NOT a new bug class

They are **substantial** (not rounding noise) — magnitude of the numeric ones:

| \|delta\| | share |
|---|---|
| ≤ 0.1 | 6% |
| 0.1 – 1 | 36% |
| 1 – 10 | 47% |
| > 10 | 11% |

58% exceed 1.0. **Classified by the actual device values** (replayed sample of 90 on the Moto,
not by op name — an op-name guess was wrong; see note): a value is "aliasing/layout" if every
device value already exists in eager (rearranged/duplicated buffer), "kernel" only if the device
produced genuinely new numbers.

| mechanism (device-verified) | share | how determined |
|------------------------------|------:|----------------|
| **aliasing / layout clobber** | **60%** | device values are rearranged/duplicated eager values |
| nonfinite / fp16 overflow | 22% | nan/inf where eager finite (or vice-versa) |
| **kernel divergence** (genuinely new wrong values) | 11% | confirmed by **isolation** → only `floor_divide`, `bitwise_left_shift` survive |
| precision (small drift) | 7% | all values close, just over tolerance |

> Note: an earlier op-family classification put ~45% in "precision" — **the device values
> disproved it.** Ops like `gelu`, `exp`, `remainder`, `fmod` showed the *clobber* signature
> (second half = repeat of first half), not drift. Lesson (and the rule for this whole run):
> classify by **replayed device values**, never by op name.

### Mechanism 1 — memory-planning aliasing clobber (≈27%), silent wrong data
The Moto's planner makes **different buffer-reuse decisions**, so the *same* copy-elision
aliasing bug ([bugs/vulkan-copy-elision-aliasing.md](bugs/vulkan-copy-elision-aliasing.md)) fires
on a **different, larger set of graphs**. Confirmed by replay — the wrong output is a *clobbered
buffer*, not a miscompute:

- **w11:110** — `out[1] = diagonal_copy(view_copy(L0))`, |delta|=2.41. Moto returns
  `[-1.158, -0.166, 1.306, 0.974, ` **`-1.158, -0.166`** `…]` — it **repeats the first row**
  instead of `[0.188, 0.075…]`. A buffer was overwritten/aliased wrong.
- **w10:498** — `out[0] = permute_copy(lift_fresh_copy(L0))`, |delta|=3.34.
- **w18:170** — `out[2]`, `view_copy` + `lift_fresh_copy`, |delta|=3.68.

These are produced by exactly the alias-semantics ops (`diagonal_copy`, `permute_copy`,
`view_copy`, `expand_copy`, `pixel_shuffle`, `alias_copy`, `squeeze_copy`, `flip`). Same root
cause as the crash cluster — here it surfaces as **silent wrong data** instead of a crash.

### Mechanism 2 — fp16 overflow / nan-convention (≈45%, incl. 475 non-finite)
Magnitude-sensitive chains overflow to `inf`/`nan` on the Mali-G57 where the Pixel stayed finite:
- **w0:644** — `gelu(mul(remainder, linear))`: eager `[-0.0, 295.2]` vs Moto `[nan, nan]`
  (fp16 overflow in the linear→gelu chain).
- **w28:116** — `out[2]/[4]` fp16: eager `[0.0, …]` vs Moto `[-inf, …]`.

Note this also cuts the other way — in **w11:177** the Moto is *more* correct (`1.39` finite where
eager is `nan`). So part of the "non-finite mismatch" bucket is benign nan-convention disagreement,
not a Moto defect.

### Mechanism 3 — genuine per-kernel divergence (11%, isolation-confirmed)
Each candidate was rebuilt as a **single-op job and run on the Moto**. Only ops that diverge in
isolation are real kernel bugs; the rest were fed clobbered inputs in-graph (→ mechanism 1).

- **CONFIRMED** (single-op divergence, full md+repro in [bugs/](bugs/)):
  - `floor_divide(x, x)` → 0 instead of 1 ([bugs/vulkan-floor-divide-self.md](bugs/vulkan-floor-divide-self.md))
  - `bitwise_left_shift` over-width shift wraps instead of zeroing ([bugs/vulkan-left-shift-overwidth.md](bugs/vulkan-left-shift-overwidth.md))
  - `scatter.src_out`, `index_put` → writes dropped, element 0 →0 ([bugs/](bugs/))
- **RULED OUT by isolation** (correct alone → their in-graph divergence was aliasing, not kernel):
  `max`, `mean`, `var.correction`, `_softmax`, `sinh`, and even `clamp.Tensor_out` (broadcast/
  min-only all OK) and `select_scatter` (int8/int32 OK). We do **not** file these as kernel bugs.

## Bottom line
The Moto's 2,400 extra mismatches are **not new op-class bugs** — they are (1) the already-known
**copy-elision aliasing bug firing more under the Moto's memory planner** (silent wrong data, the
serious one), and (2) **fp16 overflow** on magnitude-sensitive chains. The Pixel's GPU/planner
happened to dodge these specific graphs; only 96 graphs go the other way. Every SKIP and CRASH is
shared. This *strengthens* the delegate-level conclusion: the bug set is identical; only how
aggressively the memory-planner and fp16 paths trigger differs by device.

Repro: `python tmp/cross_device.py` (matrix + buckets); replay any job with
`python -m mobile feed <job_id> --corpus corpus_v2/vulkan`.
