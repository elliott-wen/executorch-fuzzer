# XNNPACK single-operator differential-fuzz — findings

Run of the **single-operator** XNNPACK corpus (`corpus_v3/xnnpack`, `--nodes 1`, **with non-finite
leaf injection**) through `xnnpack_client` on the x86 host runtime, diffed against eager PyTorch.
Methodology: [`analysis_single.md`](../../analysis_single.md) — per-operator triage, no bisection,
filter by delegated-vs-portable, name the mechanism, device-verify.

> **Run size.** Injected corpus, **87,823** graphs fed (target 100k). Supersedes an earlier
> 66,268-graph run on a *pre-injection* corpus (which missed all non-finite-input bugs).

## Verdicts

| verdict | count | share |
|---|---:|---:|
| OK | 82,357 | 93.8% |
| MISMATCH | 2,358 | 2.7% |
| CRASH | 61 | 0.07% |
| SKIP | 3,047 | 3.5% |

## The one-line story
Every graph is one op, so each non-OK outcome is already localized — no bisection. Filtering by the
delegation breakdown (`delegated.ops`): **449 of 2,358 mismatches ran on the XNNPACK delegate** (the
rest fell back to portable and are out of scope for this backend). The **non-finite leaf injection**
surfaced a family of activation-domain bugs the earlier finite-only run entirely missed.

## Confirmed XNNPACK-delegate bugs (device-verified, all with repros)

| bug | mode | mechanism | count | file |
|---|---|---|---:|---|
| **`sqrt` domain** — `sqrt(neg)=-0.0`, `sqrt(inf)=NaN` | MISMATCH | fast-rsqrt, no domain guard | 353 | [bugs/xnnpack-sqrt-domain.md](bugs/xnnpack-sqrt-domain.md) |
| **`gelu`** — `gelu(inf)=inf` not `NaN` | MISMATCH | no non-finite guard | 92 | [bugs/xnnpack-gelu-nonfinite.md](bugs/xnnpack-gelu-nonfinite.md) |
| **`relu`/`hardtanh` NaN-launder** — `NaN`→clamp bound | MISMATCH | fp16 SIMD `fmax`/`fmin` returns non-NaN operand | 4 | [bugs/xnnpack-activation-nonfinite.md](bugs/xnnpack-activation-nonfinite.md) |
| **`pixel_shuffle`/`pixel_unshuffle` load-failure** | SKIP (load) | partitioner over-inclusion, rank-7 > `XNN_MAX_TENSOR_DIMS=6` | 99 | [bugs/xnnpack-pixel-shuffle-load-failure.md](bugs/xnnpack-pixel-shuffle-load-failure.md) |
| **`_softmax` execute-failure** | SKIP (execute) | partition-vs-runtime shape mismatch (`xnn_status_invalid_parameter`) | 33 | [bugs/xnnpack-softmax-skip.md](bugs/xnnpack-softmax-skip.md) |

SKIP + CRASH reason tables (XNN-delegate vs portable-fallback split): [skips.md](skips.md).
Ruled-out suspects (portable-fallback mismatches + INT64 confounds): [ruled_out/](ruled_out/).

---

## What the injection changed (vs the pre-injection run)

The finite-only run caught only `sqrt` (neg→-0) and the `pixel_shuffle` load-failure. Adding non-finite
leaf injection **closed part of the gap** the earlier run flagged:

- ✅ **`gelu` (92)** — now caught (`gelu(inf)`); needs an `inf` input the finite corpus never produced.
- ✅ **`relu`/`hardtanh` NaN-launder (4)** — now caught; the exact clamp/min-max NaN-launder family
  that was previously missed. Confirmed only on the **fp16** SIMD path (fp32 propagates NaN).
- ✅ **`sqrt` 92 → 353** — injection adds `inf`/`±0` inputs on top of the always-present negatives.

## Why `exp` / `rsqrt` / `minimum` / `maximum` still show 0 mismatches (they are NOT out of scope)

These ops **are delegated to xnnpack** (`ops≥1` in most samples — an earlier note calling them
"portable fallback" was wrong). They show 0 mismatches because the injection picks a **uniform random**
boundary value from `{NaN, inf, -inf, 0, -0}`, and **each op's bug triggers on only one specific
value**, which the corpus didn't happen to draw at delegated samples:

| op | bug triggers on | corpus delegated+non-finite samples got | result |
|---|---|---|---|
| `exp` | **`NaN`** (`exp(NaN)=0`) | `inf` → `exp(inf)=inf` | OK |
| `rsqrt` | **`±0`** (`rsqrt(0)=inf` vs `NaN`) | `inf` → `rsqrt(inf)=0` | OK |
| `minimum`/`maximum` | **`NaN` in the winning operand** | `inf` / non-winning operand | OK |

Hand-crafted tests with the *exact* triggering value **do** mismatch on xnnpack — so the bugs are
real; the corpus just didn't sample the trigger. **These are now filed as confirmed bugs** (with
direct-trigger repros), even though the corpus reports ~0 hits:

| operator | eager → device | file |
|---|---|---|
| `exp` | `exp(NaN)`: NaN → 0.0 | [bugs/xnnpack-exp-nonfinite.md](bugs/xnnpack-exp-nonfinite.md) |
| `rsqrt` | `rsqrt(0)`: inf → NaN | [bugs/xnnpack-rsqrt-nonfinite.md](bugs/xnnpack-rsqrt-nonfinite.md) |
| `minimum` / `maximum` | `min/max(NaN,x)`: NaN → x | [bugs/xnnpack-minmax-nan-launder.md](bugs/xnnpack-minmax-nan-launder.md) |

Effective corpus trigger rate ≈ `5% (leaf injected) × 1/5 (right value) × (right operand/sign)` ≈
**≤1%**, so value-specific bugs need a much larger corpus **or a smarter injection** that biases
toward the value each domain op is sensitive to (always try `NaN`, `±0`, negative) rather than a
uniform random pick. **Recommended follow-up.**

## Reproduce
```
python -m mobile broker --job-port 15564 --client-port 15565 --ctrl-port 15566
python mobile/xnnpack_client/xnnpack_client.py --host 127.0.0.1 --client-port 15565 --ctrl-port 15566
python -m mobile feed --corpus corpus_v3/xnnpack --skip-log <fresh.tsv> \
    --host 127.0.0.1 --job-port 15564 --ctrl-port 15566 --window 64
BROKER_JOB_PORT=15564 python findings/xnnpack_x64/bugs/repro_xnnpack-gelu-nonfinite.py
```
Working data: [_work/skip_inj.tsv](_work/skip_inj.tsv) (injected run, 5,466 non-OK rows);
[_work/skip.tsv](_work/skip.tsv) (earlier pre-injection run).
