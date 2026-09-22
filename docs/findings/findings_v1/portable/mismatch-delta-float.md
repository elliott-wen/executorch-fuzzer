# MISMATCH cluster: float-tolerance value deltas (rtol>0), ~2,177

> **Per-bug splits (each with a runnable replay):** M1 negative-shift-into-float →
> [bugs/bitwise-shift-negative.md](bugs/bitwise-shift-negative.md); M2 by-zero/sign-edge →
> [bugs/remainder-sign.md](bugs/remainder-sign.md) + [bugs/floor_divide.md](bugs/floor_divide.md);
> M3/M4 benign transcendental + fp16 precision →
> [bugs/transcendental-fp16-precision.md](bugs/transcendental-fp16-precision.md).
> Full index: [REPLAY.md](REPLAY.md).

Cluster definition: `status=MISMATCH`, reason `out[i] max|delta|=<v> (rtol=0.01 atol=0.001)`.
Exact count in TSV: **2,185** rows. Sub-buckets by `<v>`: huge `>=1e15` (657), moderate `1..1e6`
(1,098), small `<1` (430).

Harness: `tmp/run_portable/cmp_deltaf.py` (adapted from `cmp_nonfinite.py`) — runs the stored
`.pte` via `et_runner.run_pte`, runs eager `g(*inputs)` on the **same** stored inputs, and prints
the portable value, eager value, |delta| and relative error at the max-delta element, plus per-output
dtypes. Op tally: `tmp/run_portable/tally.py` (maps each flagged `out[i]` to the i-th starred
returned node in the op-chain).

## Summary (headline)

**This cluster is NOT dominated by fp16 precision. The plurality is a genuine portable kernel
defect.** The single largest mechanism is `bitwise_left_shift` (and other integer/bitwise ops) with
a **negative shift amount written into a float-declared `out=` tensor**: eager produces `0`/small
values, portable produces integer magic numbers (powers of two 2^56..2^63, with 2^63 = INT64_MAX/MIN
most common). 78% of the huge bucket (515/657) are exactly power-of-two magnitudes — the fingerprint
of an integer result reinterpreted/overflowed into a float output, not floating-point overflow.

Rough mechanism split of the whole cluster:

| Mechanism | Est. share | Verdict |
|---|---|---|
| Integer/bitwise op (esp. `bitwise_left_shift` neg shift) → float `out` | ~30-40% | **Real portable bug** |
| `remainder`/`fmod`/`div` by 0 or sign-edge (NaN/inf/sign handling) | ~15% | Real bug / spec divergence |
| Transcendental approx (`erf gelu tanh tan exp sin cos atan2 asinh sigmoid rsqrt`) in fp16/fp32 | ~30% | Mostly benign precision; a few large-rel near singular points |
| fp16 reductions / small-ULP rounding exceeding `atol=0.001` | ~15% | **Benign — tolerance too strict for fp16** |

So: real bug ≈ half the cluster (the integer-into-float and divide-by-zero families), benign
precision ≈ the other half (transcendental + fp16 rounding), with the benign share concentrated in
the small bucket but contaminated by a few mis-bucketed real-bug cases.

## Dominant ops & dtypes (per sub-bucket, from `tally.py`)

Huge `>=1e15` (n=657):

| count | returned op |
|---|---|
| 326 | `bitwise_left_shift.Tensor_Scalar_out` |
| 36 | `bitwise_left_shift.Tensor_out` |
| 22 | `pow.Tensor_Scalar_out` |
| 10 | `rsub.Scalar` |
| 9 | `max.unary_out` |
| 8 | `pow.Tensor_Tensor_out` |
| 7 | `mean` / 7 `div.Scalar` / 6 `slice_scatter` / 6 `neg.out` / 6 `mul.Scalar` / 6 `minimum.out` |

`bitwise_left_shift.*` alone = 362/657 (55%). Power-of-two magnitude check: 515/657 (78%) are within
6% of an exact 2^k, k in {56..63}; top exponent 63 (×183) = INT64 saturation.

Moderate `1..1e6` (n=1,098):

| count | returned op |
|---|---|
| 220 | `remainder.Scalar_out` |
| 37 | `remainder.Tensor_out` |
| 35 | `max_pool2d_with_indices_backward.grad_input` |
| 24 | `atan2.out` |
| 22 | `floor_divide.out` |
| 21 | `erf.out` / 21 `tanh.out` / 19 `tan.out` / 19 `exp.out` |
| 18 | `bitwise_right_shift.Tensor_Scalar_out` / 18 `sub.out` / 16 `asinh.out` |

`remainder.*` = 257 (23%); the integer-into-float shift bug also re-appears here after the magic
number is `minimum`/`clamp`-ed down into the 1..1e6 band (e.g. w0:192 below).

Small `<1` (n=430): almost all transcendental/elementwise float ops — `sigmoid` 21, `remainder.Tensor`
18, `sin` 18, `fmod.Tensor` 15, `atan2` 14, `cos` 14, `tanh` 11, `asinh` 11, `div.Scalar` 11,
`tan` 11, `mean.dtype` 9, `fmod.Scalar` 9. Output dtypes here are predominantly `float16`/`float32`.

## Mechanisms (with reproductions)

### M1 — `bitwise_left_shift` with NEGATIVE shift written to a float `out=` (REAL BUG)
Eager defines left-shift by a negative amount as `0`. Portable produces an integer magic number.

- `w0:756` (`corpus/portable/w0/w0_756.py`), n0 = `bitwise_left_shift.Tensor_Scalar_out(L0, -4, out=float32)`,
  feeding out[0]=`mean.dtype_out`:
  `out[0] p=float32 portable=5.764607523034235e+17 eager=0.0` (5.76e17 = 2^59).
- `w1:1255` n3 = `bitwise_left_shift.Tensor_Scalar_out(n0, -5, out=float32)`:
  `out[1] portable=5.764607523034235e+17 eager=0.0`.
- `w0:1264` n11 = `bitwise_left_shift.Tensor_Scalar_out(n3, -7, out=float32)`:
  `out[3] portable=1.4411518807585587e+17 eager=0.0` (= 2^57).
- `w0:383` n16 = `bitwise_left_shift.Tensor_Scalar_out(n1, -6, out=float32)`:
  `out[3] portable=-1.152921504606847e+18 eager=0.0` (= -2^60).
- `w1:1363` `out[1] portable=-4.611686018427388e+19 eager=0.0` (= -2^62).
- Propagated into moderate/small bucket: `w0:192` n7 = `bitwise_left_shift...(-2, out=float32)`
  then n8 = `minimum(n7,L2)` → `out[0] portable=1.1635862588882446 eager=0.0` (magic shifted then
  clamped). `w1:673 out[1] portable=-9.223372036854776e+18 eager=0.0` (INT64_MIN).
- Direct isolation (`tmp/run_portable/iso_shift.py`): eager `bitwise_left_shift([1,2], -4)` = `[0,0]`.
  (Standalone ET export rejects the op as non-core, so it can only be reproduced through the fuzzer's
  prebuilt `.pte`, which is exactly what the corpus stores.)

Note: this is undefined-behavior-shaped — a negative/over-wide C++ shift, then an int->float store —
so the portable result is essentially garbage. The same family includes `bitwise_left_shift.Tensor_out`
(36) and `bitwise_right_shift` (18, moderate).

### M2 — `remainder`/`fmod`/`div` by zero or at sign edges (REAL BUG / spec divergence)
- `w0:1143` (`corpus/portable/w0/w0_1143.py`), the TSV-flagged out is out[2] = `cumsum.out(n8,...)`:
  `portable=2.0 eager=0.0` (|d|=2, matches `max|delta|=2.000e+00`). The same job also shows
  `remainder.Scalar(n0,-2)` and several int64 logic disagreements (out[5/6/7]) not float-precision.
- `w0:141` n14 = `remainder.Scalar(n7, 0)`; eager `remainder.Scalar(x,0.0)` = `nan` (verified in
  `iso2.py`), but integer `remainder %0` raises `ZeroDivisionError` in eager — portable instead
  returns a finite wrong value, e.g. out[2]=`_pdist_forward` `portable=3.0 eager=1.0`.
  `remainder.Scalar_out` (220) is the single biggest op in the moderate bucket.

### M3 — Transcendental approximation differences (MOSTLY BENIGN, some large-rel)
fp16/fp32 elementwise math where portable's polynomial/approx differs from eager:
- Benign: `w1:155` out[8] (fp16 scalar) `portable=0.775390625 eager=0.7890625` |d|=0.0137 rel=1.7%.
  `w11:1536` rel=1.2%, `w12:110` rel=2.1%, `w1:315` rel=6.6%.
- Larger-rel near small/singular values (still arguably approx, not garbage): `w0:1278` rel=16%,
  `w11:176` rel=29%, `w10:258` rel=39%, `w1:190` rel=58% (value 0.0027 vs 0.0063 — tiny magnitudes
  where fp16 has ~no significant bits).

### M4 — fp16 reductions / small-ULP rounding (BENIGN — `atol=0.001` too strict for fp16)
Near magnitude ~1, one fp16 ULP is ~5e-4 and an accumulated reduction easily exceeds `atol=0.001`.
These are the cleanest "tolerance-too-strict" cases (subset of small bucket with rel < ~3%).

### What the huge bucket is NOT
It is **not** fp16 overflow. Confirmed by: (a) 78% land on exact int powers of two 2^56..2^63, which
fp16 cannot even represent (fp16 max ≈ 65504); (b) the flagged outputs are declared `float32`/`int64`,
fed by integer/bitwise ops; (c) eager produces `0`, not a large finite number, so there is no
"accumulation overflow" story — it's an integer result mis-stored into the float output.

## Classification & severity

| Mechanism | Bucket(s) | Class | Severity |
|---|---|---|---|
| M1 integer/bitwise (neg-shift) → float out | huge (≈55%), some moderate/small | **Real portable bug** | High — wrong by ~1e18, UB-shaped |
| M2 remainder/fmod/div by 0 / sign edge | moderate (remainder ≈23%) | Real bug / undefined-input divergence | Medium — wrong but on degenerate inputs |
| M3 transcendental approx | small + moderate transcendentals | Benign precision (a few large-rel) | Low |
| M4 fp16 reduction rounding | small | Benign — tolerance artifact | None (harness) |

**Benign-precision share:** roughly the small bucket minus its mis-bucketed M1 cases, plus the
transcendental rows of the moderate bucket — on the order of **40-45%** of the cluster is benign
fp16/fp32 precision or formula-order rounding that merely exceeds the strict `atol=0.001`. The
remaining **~55%** (huge bucket + remainder family + propagated shifts) are real
correctness/UB defects.

## Recommendation

1. **Fix the kernels, don't loosen tolerance for these.** The huge bucket and remainder family are
   real defects. Priority: `bitwise_left_shift.Tensor_Scalar_out` / `.Tensor_out` (and `right_shift`)
   — guard negative/over-wide shift counts and the int->float `out` store to match eager
   (left-shift by negative → 0). Then `remainder.Scalar_out` divide-by-zero behavior.
2. **Scale the harness tolerance by output dtype** to remove the benign M3/M4 noise: for `float16`
   outputs use `atol≈2e-3 .. 5e-3` (≈ a few fp16 ULP at the output magnitude) or, better, a relative
   ULP-based tolerance (`atol = 4 * ulp(dtype) * max(|out|)`), and keep transcendental ops on a
   looser band. This would clear most of the small bucket without masking the integer-into-float bug
   (whose |delta| ≈ 1e18 dwarfs any dtype-scaled tolerance).
3. **Bucket by relative error, not absolute |delta|.** Absolute magnitude mis-sorts cases: real M1
   bugs whose eager value is ~0 land in the "small <1" bucket (e.g. w0:983, w1:673, w12:11 with
   rel ~1e11) while harmless fp16 rounding can show |delta| up to ~1. Triage on rel-error + dtype.

## Cited jobs
Huge/M1: w0:756, w1:1255, w0:1264, w0:383, w1:1363. Moderate/M2: w0:1143, w0:141. Propagated M1:
w0:192, w1:673. Benign M3/M4: w1:155, w11:1536, w12:110, w1:315. Larger-rel M3: w0:1278, w11:176,
w10:258, w1:190. Mis-bucketed M1 in small: w0:983, w12:11.
