# Portable-kernel bugs — per-operator index

Every entry is device-verified (portable `.pte` run on the ExecuTorch host runtime, diffed vs ATen
eager). Corpus: single-operator `corpus_v3/portable` (`--nodes 1`, with non-finite leaf injection).

Counts from the 87,921-graph run.

## MISMATCH — non-finite domain (surfaced by leaf injection)
| operator | count | eager → device | file |
|---|---:|---|---|
| `gelu` | 369 | `gelu(inf)`: NaN → inf | [gelu.md](gelu.md) |
| `_log_softmax` | 128 | non-finite: NaN → 0.0 | [log_softmax.md](log_softmax.md) |
| `grid_sampler_2d` | 102 | inf → NaN | [grid_sampler_2d.md](grid_sampler_2d.md) |
| `native_group_norm` | 29 | NaN → mixed NaN/0.0 (diff positions) | [native_group_norm.md](native_group_norm.md) |
| `addmm` | 23 | finite → all-NaN | [addmm.md](addmm.md) |

## MISMATCH — finite-value bugs
| operator | count | eager → device | file |
|---|---:|---|---|
| `remainder` / `fmod` | 247 | sign wrong: -2 → 1 | [remainder_fmod.md](remainder_fmod.md) |
| `prod` (int) | 25 | 0 → 1/-1 | [prod_int.md](prod_int.md) |
| `max_pool2d_..._backward` | 16 | one grad element wrong | [max_pool2d_backward.md](max_pool2d_backward.md) |
| `_native_batch_norm_legit.no_stats` | 44 | numerical (Δ≈4, gate N≥5) | [batch_norm.md](batch_norm.md) |

## CRASH — native abort
| operator | count | trigger | file |
|---|---:|---|---|
| `max_pool2d_..._backward` | 51 | non-finite input | [max_pool2d_backward.md](max_pool2d_backward.md) |
| `narrow_copy` / `unfold_copy` | 9 | degenerate shape / non-finite | [narrow_unfold_copy.md](narrow_unfold_copy.md) |

Ruled-out suspects (int64 confounds): [../ruled_out/](../ruled_out/). Overview: [../README.md](../README.md).
