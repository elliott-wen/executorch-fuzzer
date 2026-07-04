# Portable single-operator differential-fuzz — findings

Run of the **single-operator** portable corpus (`corpus_v3/portable`, `--nodes 1`, generated with
non-finite leaf injection) through `xnnpack_client` on the x86 host ExecuTorch runtime, diffed against
eager ATen. Methodology: [`analysis_single.md`](../../analysis_single.md). Every graph is portable
(`delegated.ops = 0`), so **portable is the backend under test** and every confirmed divergence is a
portable-kernel bug (after filtering int64 reference confounds).

> **Run size.** Latest re-feed processed **87,921** graphs (target 100k). Doubling the corpus from an
> earlier 40,150-graph pass reproduced the **same operator bug set at ~2× counts** (stability
> confirmation) and surfaced **no new confirmed portable bugs** — only additional confounds. Counts
> below are the 87,921-graph run.

## Verdicts

| verdict | count | share |
|---|---:|---|
| OK | 83,221 | 94.7% |
| MISMATCH | 1,742 | 2.0% |
| CRASH | 61 | 0.07% |
| SKIP | 2,897 | 3.3% |

**~46% of mismatches are non-finite** — surfaced by the leaf injection (inputs that a finite-only
single-op corpus would never feed the op).

## Confirmed portable-kernel bugs (per operator)

Each is device-verified with a runnable repro in [bugs/](bugs/) — full table in
[bugs/INDEX.md](bugs/INDEX.md).

### MISMATCH — non-finite input domain (injection-surfaced)
| operator | n | eager → device |
|---|--:|---|
| [`gelu`](bugs/gelu.md) | 369 | `gelu(inf)`: NaN → **inf** |
| [`_log_softmax`](bugs/log_softmax.md) | 128 | non-finite: NaN → **0.0** |
| [`grid_sampler_2d`](bugs/grid_sampler_2d.md) | 102 | inf → **NaN** |
| [`native_group_norm`](bugs/native_group_norm.md) | 29 | NaN → **mixed NaN/0.0** (diff positions) |
| [`addmm`](bugs/addmm.md) | 23 | finite → **all-NaN** |

### MISMATCH — finite-value bugs
| operator | n | eager → device |
|---|--:|---|
| [`remainder`/`fmod`](bugs/remainder_fmod.md) | 247 | sign wrong: −2 → **1** |
| [`prod` (int)](bugs/prod_int.md) | 25 | 0 → **1 / −1** |
| [`max_pool2d_…_backward`](bugs/max_pool2d_backward.md) | 16 | one gradient element wrong |
| [`_native_batch_norm_legit.no_stats`](bugs/batch_norm.md) | 44 | numerical Δ≈4 (gate N≥5) |

### CRASH — native abort
| operator | n | trigger |
|---|--:|---|
| [`max_pool2d_…_backward`](bugs/max_pool2d_backward.md) | 51 | non-finite input |
| [`narrow_copy` / `unfold_copy`](bugs/narrow_unfold_copy.md) | 9 | degenerate shape / non-finite |

### Ruled out
`bitwise_left_shift.*` (736) and `pow.*` (8) — int64 `2^63` reference sentinels + shift-UB. `any.*`
(9) — portable is mathematically correct (`any(inf)=True`); the eager reference is the anomaly on the
`out=uint8` form. `sum.IntList_out` (5), `var_mean.correction` (1) — int64/near-tolerance confounds.
None filed. See [ruled_out/](ruled_out/).

## SKIP — coverage gaps (portable rejects at runtime)
Full per-operator reason table in [skips.md](skips.md) (also has the CRASH reasons). Not bugs;
enumerated as gaps. Recurring classes: int64-index requirement (`scatter*`/`gather`), scalar-extract
guards (`arange`/`sub.Scalar`), rank/shape guards (`convolution` 5-D, `_pdist` rank≠2), unimplemented
forms (`copy` non_blocking, `expand_copy` implicit). Top: `native_group_norm` (738), `arange.start_out` (427),
`_fft_r2c` (350), `_pdist_forward` (347), `copy` (219), `expand_copy` (207, `implicit==true` not
implemented), `_log_softmax` (191), `scatter*` (~160), plus `linear`, `convolution`, `add.Scalar`,
`sub.Scalar`, `stack`. These are ops/forms with no portable kernel or a runtime guard that rejects the
graph.

## Reproduce
```
python -m mobile broker --job-port 15564 --client-port 15565 --ctrl-port 15566
python mobile/xnnpack_client/xnnpack_client.py --host 127.0.0.1 --client-port 15565 --ctrl-port 15566
python -m mobile feed --corpus corpus_v3/portable --skip-log <fresh.tsv> \
    --host 127.0.0.1 --job-port 15564 --ctrl-port 15566 --window 64
BROKER_JOB_PORT=15564 python findings/portable_x64/bugs/repro_gelu.py
```
Working data: [_work/skip2.tsv](_work/skip2.tsv) (87,921-graph run, 4,700 non-OK rows);
[_work/skip.tsv](_work/skip.tsv) (earlier 40,150-graph run).
