# Are the Pixel 9 culprits also culprits on the Moto G54?

Checked all **57** Pixel-9 culprit ops (`../vulkan_pixel9/culprits/`) against the Moto, using the
corrected metrics on both devices: **mismatch** = per-output real-bug rate (non-finite excluded),
**crash** = whole-graph crash-rate among ran graphs. Baselines: mismatch Pixel 5.1% / Moto 7.0%;
crash Pixel 9.0% / Moto 9.6%. Script: `tmp/xdev_culprits.py`.

## Answer: every Pixel culprit is present on the Moto too

No Pixel culprit disappears on the Moto. The strong ones have **near-identical rates** on both
devices → backend(delegate)-level, not device-specific. The apparent "Pixel-only" cases were
threshold/baseline artifacts (e.g. `convolution` showed "-" only because it had 143 ran-graphs on
the Moto, under the 150 cutoff — its actual crash rate is **higher** on Moto: 24.5% vs 17.1%).

### Top shared culprits — rates match across devices
| op | mismatch Pixel→Moto | crash Pixel→Moto |
|----|--------------------:|-----------------:|
| `clamp.Tensor_out` | 55.8% → 55.5% | 18.8% → 19.4% |
| `select_scatter` | 50.4% → 51.4% | 10.3% → 11.2% |
| `index_put` | 49.8% → 49.6% | 19.8% → 20.8% |
| `sum.IntList_out` | 21.8% → 21.6% | — |
| `topk.values` | 19.1% → 21.8% | — |
| `bitwise_left_shift.Tensor_out` | 21.6% → 23.5% | 11.9% → 12.6% |
| `unfold_copy` | — | 38.9% → 40.3% |
| `narrow_copy` | — | 34.7% → 36.2% |
| `scatter.src_out` | — | 34.5% → 35.6% |
| `native_layer_norm` | — | 29.6% → 30.6% |
| `clone` / `lift_fresh_copy` / `alias_copy` | — | 20–24% on both |

The entire crash cluster (copy/view/aliasing + scatter + norm) and the top mismatch culprits are
shared and within ~1 point across devices.

## Where the Moto differs: the copy/view/aliasing class fires HARDER

Several ops are **stronger or newly-culprit on the Moto**, all in the value-preserving
rearrange/copy family or magnitude-sensitive ops — consistent with the Moto's memory planner
triggering the copy-elision/aliasing bug more (see `CROSS_DEVICE_MISMATCH.md`) plus more fp16:

| op | mismatch Pixel → Moto |
|----|----------------------:|
| `pixel_shuffle` | not flagged → **26.9%** |
| `tril.out` | 4.9% → **12.7%** |
| `diagonal_copy` | not flagged → **16.5%** |
| `pixel_unshuffle` | 5.3% → 10.4% |
| `expand_copy` | 3.5% → 7.7% |
| `reflection_pad2d.out` | not flagged → 14.7% |
| `pow.Tensor_Tensor_out` | 3.3% → 7.8% |
| `any.all_out` | not flagged → 11.8% |

`convolution` crash is also higher on Moto (24.5% vs 17.1%).

## Confounds are confounds on both
The ops the Pixel pass over-blamed and per-output cleared — `_log_softmax`, `_softmax`, `tanh`,
`mean`, `rsqrt`, `logit`, `relu`, `sign` — sit at/near baseline on **both** devices (e.g.
`_softmax` 0.4% Pixel / 3.0% Moto; `logit` 0.1% / 0.2%). They are not culprits on either.

## Conclusion
The bug set is **identical across the two devices** — same operator bugs, same crash cluster, same
confounds. The only device difference is **degree**: the Moto's memory-planner/fp16 paths trigger
the copy/view aliasing class more often, raising the mismatch rate of the rearrange/copy ops. This
strengthens the delegate-level conclusion. The 57 Pixel per-op files therefore apply to the Moto
as well; the Moto adds `pixel_shuffle`, `diagonal_copy`, `reflection_pad2d`, `any.all_out` as
additional members of the same aliasing class.
