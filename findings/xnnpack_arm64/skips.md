# SKIP + CRASH reasons (xnnpack arm64 phone)

SKIP = the phone loaded the .pte but rejected it at runtime (coverage gap). Reason strings below.

| count | operator | runtime reason |
|---:|---|---|
| 716 | `native_group_norm` | ExecuTorch Error 0x12 |
| 429 | `_fft_r2c` | ExecuTorch Error 0x14 |
| 414 | `arange.start_out` | ExecuTorch Error 0x12 |
| 341 | `_pdist_forward` | ExecuTorch Error 0x12 |
| 212 | `copy` | ExecuTorch Error 0x12 |
| 202 | `expand_copy` | ExecuTorch Error 0x12 |
| 104 | `scatter_add.out` | ExecuTorch Error 0x12 |
| 81 | `scatter.src_out` | ExecuTorch Error 0x12 |
| 61 | `pixel_shuffle` | ExecuTorch Error 0x1 |
| 58 | `linear.out` | executor IllegalArgumentException: unsupported input dtype code 24 on  |
| 37 | `pixel_unshuffle` | ExecuTorch Error 0x1 |
| 32 | `scatter.value_out` | ExecuTorch Error 0x12 |
| 32 | `_softmax.out` | ExecuTorch Error 0x1 |
| 29 | `convolution` | ExecuTorch Error 0x12 |
| 9 | `stack` | ExecuTorch Error 0x12 |
| 6 | `gather.out` | ExecuTorch Error 0x12 |
| 3 | `_adaptive_avg_pool2d` | ExecuTorch Error 0x12 |
| 2 | `_native_batch_norm_legit_no_training` | ExecuTorch Error 0x12 |
| 1 | `var_mean.correction` | executor IllegalArgumentException: unsupported input dtype code 24 on  |
| 1 | `cumsum.out` | ExecuTorch Error 0x12 |

## CRASH (native abort) — corrected

The raw run (window=48) logged **2,345 CRASH**, but **2,285 were `executor unavailable`** — the phone
worker dropping under high concurrency, NOT code crashes. The full CRASH set was **re-run patiently**
(window=6, timeout=120s) and resolved cleanly in one pass (0 unavailable): **2,133 OK, 82 SKIP, 70
MISMATCH, 60 genuine native aborts** — the 60 match x86:

| count | operator | reason |
|---:|---|---|
| ~50 | `max_pool2d_with_indices_backward.grad_input` | native abort (non-finite input) |
| 5 | `unfold_copy` | native abort |
| 4 | `narrow_copy` (incl .out) | native abort |

**Methodology note:** `executor unavailable` = dropped worker, not a bug — it must be re-run, not
counted (analysis_single.md resumable-coverage rule). Here it inflated the crash count ~40×. The fix
that made it converge cleanly was **cutting in-flight concurrency** (window 48 → 6) so a brief worker
blip loses far fewer in-flight jobs.
