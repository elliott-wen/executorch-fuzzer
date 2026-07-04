# QNN (Qualcomm HTP) single-operator differential-fuzz — x86 emulator

Run of the **single-operator** Qualcomm corpus (`corpus_v3/qualcomm`, `--nodes 1`, injected) on the
**Qualcomm HTP x86 emulator** (`qnn_executor_runner` via `qnn_client`/`qnn_runner.sh`, SoC SM8450),
diffed against eager PyTorch. Methodology: [`analysis_single.md`](../../analysis_single.md). Fed via a
fleet of 20 emulator workers on an isolated broker (15564/15565/15566), ~211 graphs/s.

> **Reference caveat.** The HTP float path is **fp16** (`qualcomm.py USE_FP16`) vs the **fp32** eager
> oracle, so a fraction of mismatches are pure precision. Triage accounts for this: only ~250
> small-delta (Δ<0.05) mismatches are fp16 noise (ruled out); the findings below are far beyond it.

## Verdicts

| verdict | count | share |
|---|---:|---:|
| OK | 48,218 | 78.5% |
| MISMATCH | 7,699 | 12.5% |
| CRASH | 609 | 1.0% |
| SKIP | 4,915 | 8.0% |
| — | 61,441 | total |

Of the 7,699 mismatches: **7,047 ran on the HTP delegate** (652 portable-fallback, out of scope).

## Headline: the HTP fp16 path has no IEEE non-finite semantics

**~5,374 delegated mismatches are non-finite** — every transcendental/activation kernel returns a
wrong **finite** value (or the wrong non-finite) for an `inf`/`nan` input instead of propagating.
Device-verified:

| op | eager | QNN HTP |
|---|---|---|
| `log(inf)` | inf | **-0.000489** |
| `exp(inf)` | inf | **131008** (overflow to huge finite) |
| `cos(nan)` | nan | **1.0** |
| `atan(inf)` | π/2 | **nan** |
| `elu(inf)` | inf | **-65472** (sign flip + saturate) |
| `sqrt(neg)` | nan | **-131008** |

Signature: `inf → ±131008 / ±65472 / ~0`, `nan → plausible-but-wrong finite`. This is a whole-backend
gap (the HTP kernels never see IEEE non-finites in real models). Full list:
[bugs/qnn-nonfinite-mishandling.md](bugs/qnn-nonfinite-mishandling.md).

## Other confirmed findings
- **[`rsub.Scalar` value bug](bugs/qnn-rsub-value.md)** (~297): `rsub` returns the wrong result on
  **finite** inputs (`18 → 9`, Δ=9 ≫ fp16 tolerance) — not precision. Plus a `sub` Δ1–100 family.
- **[Crash cluster](bugs/qnn-crash-cluster.md)** (609): `qnn_executor_runner` **Aborts (444)** or
  **Segfaults (165)** — unlike the phones' opaque native abort, the emulator gives a real signal.
  Dominated by `stack` (329), then pad ops (`replication_pad3d/2d`, `reflection_pad2d`), `copy`,
  `split_with_sizes_copy`, `max_pool2d_backward`.

## SKIP — HTP unsupported ops ([skips.md](skips.md))
All skips return `qnn_runner rc=2` (graph not runnable on HTP) with no further detail; the finding is
the op list: `native_group_norm` (593), `pixel_shuffle` (400), `narrow_copy` (371), `any.dims` (336),
`_fft_r2c` (331), `_pdist_forward` (321), `pixel_unshuffle` (287), `copy` (223), `topk` (188),
`var.correction` (177), … 86 distinct ops the HTP delegate rejects.

## Takeaway
QNN HTP's dominant correctness issue is **non-finite handling** (no IEEE inf/nan) — pervasive across
transcendental/activation ops — plus a `stack`/pad/copy **crash** cluster and a `rsub` value bug. The
non-finite family was only exposed because the corpus **injects** boundary values; a finite-only corpus
would have reported the HTP as far cleaner than it is. Unlike the phones, QNN crashes carry a real
Abort/Segfault signal (see [skips.md](skips.md)).

## Reproduce
```
# broker + QNN emulator client fleet (needs QNN_SDK_ROOT via android-env.sh + x86 qnn_executor_runner)
python -m mobile broker --job-port 15564 --client-port 15565 --ctrl-port 15566 -v
source mobile/android-dev/android-env.sh
ET_BUILD_X86=mobile/pytorch_ref/executorch/build-x86 \
  python mobile/qnn_client/qnn_client.py --host 127.0.0.1 --client-port 15565 --ctrl-port 15566 --soc SM8450
python -m mobile feed --corpus corpus_v3/qualcomm --skip-log <fresh.tsv> \
    --host 127.0.0.1 --job-port 15564 --ctrl-port 15566 --window 20 --timeout 150
BROKER_JOB_PORT=15564 python findings/qnn_emulator/bugs/repro_qnn-rsub-value.py
```
Working data: [_work/skip.tsv](_work/skip.tsv).
