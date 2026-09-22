# VGF single-op SKIP / CRASH reasons

Runtime rejects (`rc=2`), per operator. **Reason is the finding** — the guard that fired at
VGF runtime. Reasons captured verbatim from the runner `run.log` (step 4).

## A. Delegated forms rejected by the VGF runtime (coverage gap / partitioner-runtime mismatch)
These ops **partition to VGF** (`delegated≥1`) but the runtime rejects them — the partitioner
claims ops it can't execute.

| count | operator | verbatim runtime reason |
|--:|---|---|
| 439 | `fill.Tensor` | Failed to process VGF blob → Init failed for VgfBackend 0x1 (blob won't compile) |
| 383 | `copy` | Failed to process VGF blob → Init 0x1 |
| 294 | `max_pool2d_with_indices_backward.grad_input` | Failed to process VGF blob → Init 0x1 |
| 264 | `any.dims_out` | Failed to allocate tensor for VGF resource 0 → Init 0x1 |
| 237 | `pixel_shuffle` | Failed to process VGF blob → Init 0x1 |
| 156 | `topk.values` | Failed to process VGF blob → Init 0x1 |
| 110 | `pixel_unshuffle` | Failed to process VGF blob → Init 0x1 |
| 108 | `where.self` | execute: Output tensor byte size 4 != IO allocation 2 (0x12) — IO dtype-width bug |
| 81 | `any.all_out` | Failed to allocate tensor for VGF resource → Init 0x1 |
| 81 | `mean.out` | Failed to process VGF blob → Init 0x1 |
| 75 | `sum.IntList_out` | Failed to process VGF blob → Init 0x1 |
| 56 | `index_select` | Failed to process VGF blob → Init 0x1 |
| 33 | `amin.out` | Failed to process VGF blob → Init 0x1 |
| 24 | `mean.dtype_out` | rc=2 (VGF blob init / execute failure; run.log for detail) |
| 22 | `amax.out` | rc=2 (VGF blob init / execute failure; run.log for detail) |
| 22 | `where.self_out` | Failed to process VGF blob → Init 0x1 |
| 17 | `index_select.out` | rc=2 (VGF blob init / execute failure; run.log for detail) |
| 9 | `prod.int_out` | rc=2 (VGF blob init / execute failure; run.log for detail) |
| 8 | `mean` | rc=2 (VGF blob init / execute failure; run.log for detail) |
| 8 | `glu.out` | rc=2 (VGF blob init / execute failure; run.log for detail) |
| 6 | `unsqueeze_copy` | rc=2 (VGF blob init / execute failure; run.log for detail) |
| 5 | `hardtanh` | rc=2 (VGF blob init / execute failure; run.log for detail) |

**Two distinct signatures:** (1) *`Failed to process VGF blob` → Init 0x1* at method-load —
the model-converter blob won't JIT-compile on the emulation layer (most ops above); (2)
*`Output tensor byte size 4 != IO allocation 2` → execute 0x12* — `where.self` sizes an IO at
the wrong dtype width (a real delegate IO-metadata bug, not just an unsupported op).

## B. Portable-fallback missing-kernel skips (NOT a VGF bug — ruled out)
These ops fell back to portable (`delegated=0`); the portable runtime lacks the kernel or the
graph fails to load. A **portable** coverage gap, filtered out of the VGF findings per step 3a.

| count | operator |
|--:|---|
| 435 | `_fft_r2c` |
| 350 | `_pdist_forward` |
| 169 | `expand_copy` |
| 120 | `scatter_add.out` |
| 89 | `scatter.src_out` |
| 80 | `linear.out` |
| 64 | `arange.start_out` |
| 52 | `linear` |
| 27 | `copy` |
| 26 | `scatter.value_out` |
| 21 | `convolution` |
| 15 | `stack` |
| 6 | `gather.out` |
| 5 | `pixel_shuffle` |

## Crash
7 native aborts total, **all in portable-fallback graphs** — zero VGF-delegate crashes. Native
aborts carry no catchable message; the crashing graphs are portable, so not a VGF finding.