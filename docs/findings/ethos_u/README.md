# Ethos-U (Corstone-300 FVP) — single-operator differential-fuzz findings

Backend: **ethos-u** (Ethos-U55 NPU delegate, Vela command stream), run on the Arm **Corstone-300
FVP** (`ethos-u55-128`) via `fvp_client`. Corpus: `corpus_v3/ethos-u` — a **single-operator** corpus
(`--nodes 1`: each graph is exactly one op wired to leaf inputs, so every non-OK outcome is already a
minimal repro attributed to one operator). Both Arm backends are int8 (`quant=ALWAYS`); the compare
is quantized-reference vs device, same quantization space. Method: `analysis_single.md`.

## Run summary
Full corpus fed through a 48-client FVP fleet, one dedicated broker; **no infra dropouts**.

| total | OK | MISMATCH | CRASH | SKIP | TIMEOUT |
|------:|---:|---------:|------:|-----:|--------:|
| 83,312 | 74,825 | 3,402 | 442 | 4,643 | **0** |

**The Ethos-U delegate surface is small:** only **9,342 jobs (11%)** actually delegated to the NPU
(`ops≥1`); **73,970 (89%)** fell back to the portable CPU kernel (`ops=0`) and are out of scope for
Ethos-U (filter 3a). Of the 1,973 **delegated** non-OK outcomes (1,590 MISMATCH + 383 SKIP), the
domain classifier splits them into 799 non-finite-input (3e), 794 negative-shift UB (3d), and **380
FINITE-CLEAN** genuine candidates → **166 MISMATCH across 7 ops** (all confirmed) + 214 SKIP.

## Confirmed Ethos-U operator bugs
All delegated (`ops≥1`), finite-input, **deterministic (REPRO 5/5)** on the FVP, mechanism derived
from device-verified `eager` vs `device` values. Repros in `bugs/`, index in [bugs/INDEX.md](bugs/INDEX.md).

### MISMATCH — WRONG-VALUE (7 operators)

| operator | mechanism | eager → device (example) | prevalence |
|----------|-----------|--------------------------|-----------|
| [`fill.Scalar`](bugs/fill-scalar.md) | fill scalar ignored; output is fixed input-derived garbage | `[1,1,1,1]` → `[-3,-1,-3,-1]` | 96/100 |
| [`rsub.Scalar`](bugs/rsub-scalar.md) | output collapses to ~0 past first element(s) (dropped stores) | `[-4.85,-5.85,-7.29,…]` → `[-4.85,0,0,…]` | 50/257 |
| [`floor_divide`](bugs/floor-divide.md) | rounding-direction off-by-one at integer boundary | `floor(1.16)=1` → `0` | 9/307 |
| [`bitwise_right_shift.Tensor_out`](bugs/bitwise-right-shift.md) | logical (not arithmetic) shift of negatives | `-3>>1 = -2` → `-1` | 6 |
| [`clamp.Tensor_out`](bugs/clamp-tensor.md) | min/max tensor broadcast fails; output = constant | `[13,0,14,…]` → `[13,13,13,…]` | 3/3 |
| [`sub.out`](bugs/sub-out.md) | scalar-broadcast subtraction wrong past first element | `[12,12,-23,…]` → `[12,15,15,…]` | 1 |
| [`remainder.Tensor_out`](bugs/remainder-tensor.md) | wrong sign for negative divisor | `1 % -3 = -2` → `1` | 1 |

Two recurring root-cause families: **broadcast handling** (`clamp.Tensor_out`, `sub.out`, and the
tail-zeroing in `rsub.Scalar` — the first element is right, the broadcast/remainder collapses) and
**integer semantics** (`bitwise_right_shift` sign-extension, `remainder` divisor sign,
`floor_divide` rounding). `fill.Scalar` is the most severe: the scalar operand is never applied.

### SKIP — Ethos-U coverage gaps (see [skips.md](skips.md))
- `stack` (25) and the constant-creation ops `zeros/ones/full/arange[.start]_out` (184) → FVP boots
  but emits **no output** (`no out-*.bin`). Constant ops are degenerate/constant-folded graphs (a
  harness limitation); `stack` emitting nothing is a genuine gap. REPRO 5/5.
- `bitwise_left/right_shift.Tensor_out` (174) → SKIP `rc=2`, fail to execute on many forms. REPRO 5/5.

### CRASH
No delegated (Ethos-U) crashes. The only CRASHes (442, all `_fft_r2c`) are **portable** and are a
runner **build** failure (rc=5), not a delegate crash — see [skips.md](skips.md). `TIMEOUT=0`.

## Per-operator table (delegated / Ethos-U surface)
Non-OK counts over each op's samples; `jobs` = delegated jobs for that op. Full table with the
portable (ruled-out) half is in `_work/` (`analyze.py`). FINITE-CLEAN = genuine, after 3d/3e.

| operator | MISM | SKIP | jobs | FINITE-CLEAN verdict |
|----------|-----:|-----:|-----:|----------------------|
| `fill.Scalar` | 96 | 0 | 100 | **BUG** (WRONG-VALUE) |
| `rsub.Scalar` | 89 | 0 | 257 | **BUG** 50 clean / 39 non-finite |
| `index_put` | 205 | 0 | 218 | non-finite-input only (ruled out) |
| `select_copy.int` | 124 | 0 | 275 | non-finite-input only (ruled out) |
| `unsqueeze_copy` | 86 | 0 | 296 | non-finite-input only (ruled out) |
| `floor_divide` | 65 | 0 | 307 | **BUG** 9 clean / 56 non-finite |
| `split_copy.Tensor` | 58 | 0 | 259 | non-finite-input only (ruled out) |
| `relu` | 42 | 0 | 229 | non-finite-input only (ruled out) |
| `div.Scalar` / `mul.Scalar` / `abs` / `permute_copy` | 33/31/28/33 | 0 | ~280 each | non-finite-input only (ruled out) |
| `clamp.Tensor_out` | 3 | 0 | 114 | **BUG** (broadcast) |
| `sub.out` / `remainder.Tensor_out` | 1 / 1 | 0 | 107 / 1 | **BUG** (broadcast / sign) |
| `bitwise_left_shift.Tensor_out` | 308 | 88 | 408 | 2 clean / 394 neg-shift UB; SKIP gap |
| `bitwise_right_shift.Tensor_out` | 323 | 86 | 409 | **BUG** 6 clean / 400 neg-shift UB; SKIP gap |
| `stack` | 0 | 25 | 25 | **SKIP gap** (no output) |
| `zeros/ones/full/arange[.start]_out` | 0 | 184 | ~184 | SKIP (degenerate constant graphs) |

## Coverage ledger (the definition of done)
Seed schedule = round-robin over **228** `EXECUTORCH_OPS`. Assertion holds:
`228 scheduled = 199 produced + 29 never-produced`, and `199 produced = 63 delegatable + 136 always-portable`.

- **199 produced ops** got a verdict; **63** ever delegate to Ethos-U (the backend surface), **136**
  always fall back to portable (0 delegated jobs, out of Ethos-U scope).
- **29 scheduled ops never lowered** (coverage gaps / attrition): `max_pool2d_with_indices_backward`
  (332 gen-crashes), the FFT family (`_fft_c2r*`, `_fft_r2c.out`, `_conj_physical`,
  `view_as_real_copy`), batch-norm legit variants, data-dependent-shape ops (`nonzero[.out]`,
  `masked_select[.out]`, `max/min.dim`, `topk.values`), and `cat.out`, `stack.out`,
  `split[_with_sizes]_copy.*out`, `unbind_copy.int_out`, `as_strided_copy`, `native_dropout`,
  `rand.out`, `repeat_interleave.Tensor`, `reflection_pad3d.out`, `convolution_backward`,
  `any.out`. Full list in `_work/coverage_ledger.txt`.
- Generation-time `_crashes`: 723 graphs over 19 ops (mostly `max_pool2d_backward` 332, `repeat` 282).

## What this corpus can and cannot show
Single-op graphs isolate each finding to one operator (incl. its own lowering), but by construction
**cannot** surface cross-op bugs (memory planning, buffer aliasing, cross-op fusion, dtype/layout
rewrites) — no siblings, no multi-node plan. Non-finite-**input** propagation is likewise a boundary
of `--nodes 1` (surfaced as the 799 ruled-out 3e cases). Every confirmed bug here is attributable to
one Ethos-U delegate kernel on finite inputs.

## Reproduce / rebuild
```
# 1. broker + FVP fleet (the exact lane this corpus was fed through)
bash findings/ethos_u/_work/start_broker.sh
bash findings/ethos_u/_work/start_fleet.sh 8 0          # 8 clients is plenty for repros
# 2. any confirmed bug
python findings/ethos_u/bugs/repro_fill-scalar.py
# 3. rebuild the analysis from skip.tsv
python findings/ethos_u/_work/analyze.py        # per-op table (delegated vs portable)
python findings/ethos_u/_work/classify.py       # 3d/3e domain classification -> FINITE-CLEAN worklist
python findings/ethos_u/_work/skips_report.py    # SKIP/CRASH reason signatures
PYTHONPATH=/data/jwen929 python findings/ethos_u/_work/coverage.py   # coverage ledger
python findings/ethos_u/_work/determinism.py 5 <job_id> …            # 3b determinism gate
```

## Artifacts
- `bugs/` — 7 confirmed operator bugs (md + runnable one-op repro), [bugs/INDEX.md](bugs/INDEX.md).
- `skips.md` — per-operator SKIP + CRASH reason tables (verbatim runtime strings).
- `ruled_out/` — suspects removed by filters 3a (portable), 3e (non-finite), 3d (negative-shift UB).
- `_work/` — raw skip-log (`skip.tsv`), manifest, feed log, coverage ledger, and all analysis tools.
