# Bug: XNNPACK clamp / min / max / relu / hardtanh launder NaN to a finite value

> **Classification:** real XNNPACK-delegate **correctness** divergence (silent loss of the NaN
> signal). Reproduces on the x86 host (and ARM). The portable backend handles all five ops
> correctly, so it only bites when the XNNPACK delegate is active.
>
> Split out of the former `xnnpack-activation-nonfinite.md` (Mechanism A). Sibling mechanisms:
> [sqrt/rsqrt domain](xnnpack-sqrt-rsqrt-domain.md), [exp NaN/overflow](xnnpack-exp-nonfinite.md).
> Buggy model graph: [`graph_xnnpack-clamp-minmax-nan.py`](graph_xnnpack-clamp-minmax-nan.py).

## Headline

`relu`, `clamp`, `hardtanh`, `minimum` and `maximum` produce a **finite, wrong** result where
eager PyTorch produces **NaN**, *only* under the XNNPACK delegate. All five share **one root**:
they lower to XNNPACK clamp/min/max nodes whose microkernels use IEEE `fmin`/`fmax`-style min/max
that **returns the non-NaN operand** — so `f(NaN)` returns the clamp bound (or the other operand)
instead of `NaN`. Eager ATen `relu`/`clamp`/`hardtanh`/`minimum`/`maximum` all **propagate NaN**.

## Born-here evidence (eager vs xnnpack at the divergent element)

| op | job_id | node | eager `e[0]` | xnnpack `b[0]` | note |
|----|--------|------|------|------|------|
| hardtanh | `w0:49`   | n17 | `nan` | `-2` | clamp lo bound |
| hardtanh | `w0:570`  | n3  | `nan` | `3`  | clamp hi bound |
| hardtanh | `w1:756`  | n13 | `nan` | `-5` | clamp lo bound |
| hardtanh | `w10:116` | n5  | `nan` | `-7` | clamp lo bound |
| hardtanh | `w10:1566`| n19 | `nan` | `1`  | clamp hi bound |
| relu     | `w1:1017` | n11 | `nan` | `0`  | relu(nan)=max(nan,0)=0 |
| relu     | `w10:21`  | n19 | `nan` | `0`  | relu(nan)=0 |
| minimum  | `w0:734`  | n9  | `nan` | `1`  | min(nan,1)=other operand |
| minimum  | `w0:1037` | n17 | `nan` | `1`  | min(nan,1)=1 |
| minimum  | `w11:426` | n12 | `nan` | `0`  | min(nan,0)=0 |
| maximum  | `w10:385` | n17 | `nan` | `1`  | max(nan,1)=1 |

Pasted localizer output (leads):

```
corpus/xnnpack/w0/w0_49.py  -> Divergence(node='n17', op='hardtanh', kind='nonfinite', index=17) e[0]=nan b[0]=-2
corpus/xnnpack/w0/w0_734.py -> Divergence(node='n9',  op='minimum',  kind='nonfinite', index=9)  e[0]=nan b[0]=1
```

Localizer: `PYTHONPATH=/data/jwen929 QNN_SDK_ROOT="" .venv/bin/python -m mobile.gen.divergence --backend=xnnpack <py>`.

## Root cause — all five ops lower to the SAME clamp/min-max kernel family

`relu`, `clamp`, `hardtanh` all serialize to a single **`XNNClamp`** node with `OutputMinMax`
bounds (relu = clamp(0, +inf); hardtanh = clamp(lo, hi); clamp = clamp(min, max)):

- `pytorch_ref/executorch/backends/xnnpack/operators/op_relu.py:50-60`
  (`OutputMinMax(output_min=0, output_max="+inf")` → `XNNClamp`)
- `pytorch_ref/executorch/backends/xnnpack/operators/op_clamp.py:38-63` (min_val/max_val → `XNNClamp`)
- `pytorch_ref/executorch/backends/xnnpack/operators/op_hardtanh.py:39-66` (output_min/output_max → `XNNClamp`)

`minimum`/`maximum` serialize to `XNNMinimum`/`XNNMaximum` (vbinary vmin/vmax):

- `pytorch_ref/executorch/backends/xnnpack/operators/op_minimum.py` → `XNNMinimum`
- `pytorch_ref/executorch/backends/xnnpack/operators/op_maximum.py` → `XNNMaximum`

The clamp microkernel applies the bounds with `max` then `min`, **bound as first operand**:

- `.../third-party/XNNPACK/src/f32-vclamp/gen/f32-vclamp-scalar.c:40-41`
  ```c
  vacc = xnn_max_f32(vmin, vacc);   // lower-bound the data
  vacc = xnn_min_f32(vmax, vacc);   // upper-bound the data
  ```

`xnn_min_f32`/`xnn_max_f32` and the vmin/vmax binary kernels all resolve to NaN-laundering min/max:

- **x86 SSE/AVX (host fuzzing platform):** `_mm_max_ps`/`_mm_min_ps` (and AVX `_mm256_*`) implement
  `cmp ? a : b` semantics that return the **second** operand on unordered compares —
  `.../src/xnnpack/simd/f32-sse2-base.h:55-63`, `.../src/xnnpack/simd/f32-avx-base.h:68-76`.
- **ARM v8 / NEON:** `math_min_f32`/`math_max_f32` → `__builtin_fminf`/`__builtin_fmaxf` (IEEE-754-2008
  `fmin`/`fmax`, returns the **non-NaN** operand) — `.../src/xnnpack/math.h:244-261`;
  `vmaxq_f32`/`vminq_f32` — `.../src/xnnpack/simd/f32-neon.h:110-117`.
- **Scalar fallback** `a > b ? a : b` — `.../src/xnnpack/simd/f32-scalar.h:75-83`: for
  `xnn_max_f32(vmin, vacc)` with `vacc=NaN`, `vmin > NaN` is false → returns `vacc=NaN`; the binary
  `math_*_f32` `XNN_UNPREDICTABLE(b < a) ? ...` path returns the **non-NaN** operand.

**Net effect:** the clamp bound (or the other tensor element for min/max) is returned; the NaN is
laundered to a finite value. One kernel pattern (min/max that does not propagate NaN), reached by
all five ops.

### Eager (ATen) semantics
`relu`/`clamp`/`hardtanh`/`minimum`/`maximum` **propagate NaN**: any NaN operand yields NaN
(e.g. `torch.minimum(nan, 1) = nan`, `relu(nan) = nan`).

## Scope

- Per-op non-finite mismatch propensity (xnnpack vs **portable** in parens) — all introduced by the
  delegate: `relu` 3.7% (0.3%), `clamp` 2.4%, `hardtanh` 4.0%, `minimum` 3.9%, `maximum` 3.1%.
- A **5–14× spike** over portable; portable's clamp/min/max propagate NaN correctly. Combined with
  the sqrt/exp mechanisms this cluster accounts for thousands of the 4,808 `non-finite` MISMATCH
  rows in `tmp/run_xnnpack/skip_reasons_xnnpack.tsv`.

## Buggy model graph

[`graph_xnnpack-clamp-minmax-nan.py`](graph_xnnpack-clamp-minmax-nan.py) — the representative
corpus graph `w0:49`. Run it to lower through XNNPACK and print the first born-here divergence
(`hardtanh` n17, `eager=nan` vs `xnnpack=-2`):

```bash
.venv/bin/python findings/xnnpack_x64/bugs/graph_xnnpack-clamp-minmax-nan.py
# expected: Divergence(node='n17', op='hardtanh', kind='nonfinite') e[0]=nan b[0]=-2
```

This reproduces in-process on the **x86 host** via the first-divergence localizer (no phone/worker
needed). **Not re-run here** — no xnnpack host client was available this round; the expected line
is the documented localizer result.

## Fix / recommendation

Make clamp/min/max NaN-propagating, or do not partition these ops to XNNPACK, in order of preference:

1. In ExecuTorch, **decline to partition** `XNNClamp`/`XNNMinimum`/`XNNMaximum` (relu/clamp/hardtanh/
   minimum/maximum) so they fall back to the already-correct portable (NaN-propagating) kernel —
   lowest-risk, especially under a strict-IEEE mode.
2. Upstream-XNNPACK: provide / select **NaN-propagating** min/max microkernels for clamp (e.g. an
   explicit `cmpunord` OR-in of NaN) behind a "propagate NaN" flag, and have ExecuTorch's
   clamp/relu/hardtanh visitors set it. ATen semantics require propagation.
3. At minimum, **document** that XNNPACK-delegated `relu/clamp/hardtanh/minimum/maximum` do not
   propagate NaN.
