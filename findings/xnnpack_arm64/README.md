# XNNPACK single-operator differential-fuzz — arm64 phone

Run of the **single-operator** XNNPACK corpus (`corpus_v3/xnnpack`, `--nodes 1`, with non-finite leaf
injection) on a **real arm64 Android phone** (worker connected to the broker via a rathole tunnel;
broker on 15554/15555/15556), diffed against eager PyTorch on the host. Methodology:
[`analysis_single.md`](../../analysis_single.md). Same corpus as the x86 run
([../xnnpack_x64](../xnnpack_x64/)), so this is a clean **arm64 (NEON) vs x86 (SSE/AVX)** differential.

## Verdicts (corrected)

| verdict | count | note |
|---|---:|---|
| OK | ~82,527 | |
| MISMATCH | ~2,383 | 444 delegated (xnnpack kernel), rest portable-fallback |
| CRASH | ~60 | genuine native aborts (see below) |
| SKIP | ~2,853 | coverage gaps — [skips.md](skips.md) |
| — | 87,823 | total graphs fed |

> **Flaky-worker correction.** The raw run (window=48) logged **2,345 CRASH**, but **2,285 were
> `executor unavailable`** — the phone worker dropping under high concurrency, not code crashes. The
> full CRASH set was **re-run patiently** (window=6, timeout=120s) and resolved in a single clean pass
> (0 unavailable): **2,133 OK, 82 SKIP, 70 MISMATCH, 60 genuine native aborts**. The 70 mismatches +
> 82 skips were merged into the numbers below. Counting the dropouts as crashes would have overstated
> the arm64 crash rate ~40×; the fix was cutting in-flight concurrency, not accepting the dropouts.

## Confirmed XNNPACK-delegate bugs (device-verified on the phone)

| bug | mode | arch | count | file |
|---|---|---|---:|---|
| **`log` saturate** — `log(inf)` → ~88.38 not `inf` | MISMATCH | **arm64 only** | 180 | [bugs/xnnpack-log-saturate.md](bugs/xnnpack-log-saturate.md) |
| **`logit` saturate** — `+inf` → ~88.38 | MISMATCH | **arm64 only** | 187 | [bugs/xnnpack-logit-saturate.md](bugs/xnnpack-logit-saturate.md) |
| **`gelu`** — `gelu(inf)` → inf not `NaN` | MISMATCH | shared x86+arm64 | 92 | [bugs/xnnpack-gelu-nonfinite.md](bugs/xnnpack-gelu-nonfinite.md) |
| native-abort crashes (`max_pool2d_…_backward`, `unfold_copy`, `narrow_copy`) | CRASH | shared | ~60 | [skips.md](skips.md) |

SKIP/CRASH reason tables: [skips.md](skips.md). Ruled-out (portable-fallback + INT64 confounds): [ruled_out/](ruled_out/).

## The headline: arm64 (NEON) ≠ x86 (SSE/AVX) XNNPACK kernels

The same corpus produces a **different delegated-bug set** per architecture, because XNNPACK ships
different SIMD kernels:

| op | x86 (SSE/AVX) | arm64 (NEON) |
|---|---|---|
| `sqrt` | **buggy** — `sqrt(neg)=-0.0`, `sqrt(inf)=NaN` (353) | **correct** — 0 mismatches |
| `log` | correct | **buggy** — `log(inf)` saturates to ~88.38 (180) |
| `logit` | correct | **buggy** — `+inf` saturates to ~88.38 (187) |
| `relu`/`hardtanh` NaN-launder | buggy (fp16 SIMD, few) | not observed |
| `gelu` | buggy (`inf`→inf) | buggy (`inf`→inf) — **shared** |

**Takeaway:** a backend fuzzing pass on the host x86 runtime is **not** a substitute for the real
device — the arm64 NEON activation/transcendental kernels (`log`, `logit`) have their own non-finite
saturation bugs, and the x86 `sqrt` domain bug does not exist on arm64. Only `gelu` is truly
backend-level (both). This is exactly why the run must happen on the phone.

## Reproduce
```
python -m mobile broker --job-port 15554 --client-port 15555 --ctrl-port 15556 -v
# phone connects as worker via the rathole tunnel to client-port 15555 (human sets up)
python -m mobile feed --corpus corpus_v3/xnnpack --skip-log <fresh.tsv> \
    --host 127.0.0.1 --job-port 15554 --ctrl-port 15556 --window 48 --timeout 60
# re-feed any 'executor unavailable' rows until every job has a real verdict:
python -m mobile feed --from-tsv <skip.tsv> --status CRASH --corpus corpus_v3/xnnpack ...
BROKER_JOB_PORT=15554 python findings/xnnpack_arm64/bugs/repro_xnnpack-log-nonfinite.py
```
Working data: [_work/skip.tsv](_work/skip.tsv) (raw run), plus CRASH re-feed verdicts.
