# MediaTek (Neuron) — SKIP & CRASH reason tables

Device: MediaTek Dimensity phone worker over the broker (client-port 15555), corpus `corpus_v3/mtk`
(single-operator, `--nodes 1`; 55,756 jobs). Reasons are the verbatim runtime strings from the
feeder skip-log (`_work/skip_full.tsv`, column `reason`). **The reason is the finding for these modes.**

## SKIP — delegated (mtk coverage gaps: op partitioned into the Neuron delegate but rejected at runtime)

| count | operator | runtime reason (verbatim) |
|------:|----------|---------------------------|
| 305 | `replication_pad1d.out` | `[ExecuTorch Error 0x12] Invalid argument: Execution failed for method: forward` |
| 254 | `replication_pad3d.out` | `[ExecuTorch Error 0x12] Invalid argument: Execution failed for method: forward` |
| 236 | `replication_pad2d.out` | `[ExecuTorch Error 0x12] Invalid argument: Execution failed for method: forward` |
| 1 | `roll` | `[ExecuTorch Error 0x2] Invalid state: Execution failed for method: forward` |
| 1 | `constant_pad_nd` | `[ExecuTorch Error 0x2] Invalid state: Execution failed for method: forward` |
| 1 | `squeeze_copy.dim` | `[ExecuTorch Error 0x2] Invalid state: Execution failed for method: forward` |

The `replication_pad{1,2,3}d.out` SKIPs (795 total) are the real delegated coverage gap: these ops
pass partitioning (delegate `ops≥1`) but the Neuron runtime aborts them with **Invalid argument
(0x12)**. The generic ExecuTorch wrapper does not surface the underlying Neuron check clause. The few
forms that *do* execute return wrong values (see the Mismatch section / `bugs/replication_pad`).

## SKIP — portable (`ops=0`, NOT a mtk bug: op fell back to the portable CPU kernel)

Ruled out of mtk attribution (step 3a) but recorded as **portable-runtime coverage gaps** on this
device build:

| count | operator | runtime reason (verbatim) |
|------:|----------|---------------------------|
| 442 | `_fft_r2c` | `[ExecuTorch Error 0x14] Operator missing` (no portable kernel on device) |
| 348 | `_pdist_forward` | `[ExecuTorch Error 0x12] Invalid argument` |
| 114 | `scatter_add.out` | `[ExecuTorch Error 0x12] Invalid argument` |
| 69 | `scatter.src_out` | `[ExecuTorch Error 0x12] Invalid argument` |
| 25 | `scatter.value_out` | `[ExecuTorch Error 0x12] Invalid argument` |
| 5 | `gather.out` | `[ExecuTorch Error 0x12] Invalid argument` |
| 3 | `narrow_copy` | `[ExecuTorch Error 0x12] Invalid argument` |
| 3 | `_adaptive_avg_pool2d` | `[ExecuTorch Error 0x12] Invalid argument` |
| 1 | `cumsum.out` | `[ExecuTorch Error 0x12] Invalid argument` |
| 1 | `var.correction` | `IllegalArgumentException: unsupported input dtype code 24 on Android` (bf16?) |

(1011 portable SKIPs total. `_fft_r2c` "Operator missing" = the op has no on-device portable kernel.)

## CRASH — delegated (native abort, no catchable message)

**Only `copy` is a confirmed deterministic crash.** The bulk feed logged **4,517** CRASH rows, but a
serial (`--window 1`) re-run of that exact set returned **565 CRASH, 3,610 OK, 342 MISMATCH** — i.e.
~80% of the "crashes" were **crash-storm collateral**: when the executor process native-aborts, every
in-flight neighbour (and, in serial mode, the very next job fed before the client restarts the
executor) is logged CRASH too. A 5×-per-job determinism gate on a sample of the non-`copy` crashers
returned **OK 5/5 for every one** — they are restart artifacts, not real crashes. (Only 1 job across
the whole run was `executor unavailable`; the rest were genuine `native abort` of the process.)

| count | operator | crash trigger | status |
|------:|----------|---------------|--------|
| 217/217 | `copy` | any `aten.copy(self, src, non_blocking)` form (e.g. (2,4)←broadcast (1,4), fp32) | **CONFIRMED** — native abort, reproduces 5/5 in isolation |
| ~348 (across ~50 ops) | linear, squeeze_copy, div, pow, bmm, sigmoid, group_norm, … | logged CRASH in the serial re-run but **OK 5/5** when replayed in isolation | RULED OUT — post-abort restart collateral |

Native abort ⇒ no catchable error string (`executor process died (native abort)` only). The absence
of a guard is the bug: the delegate should reject `copy` with a catchable error rather than abort the
process. See `bugs/copy.md`.
