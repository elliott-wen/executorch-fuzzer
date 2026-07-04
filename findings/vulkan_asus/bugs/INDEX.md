# Confirmed Vulkan-delegate bugs (ASUS phone) — index

Device-verified on the phone. All delegated (`ops≥1`). Corpus: single-op `corpus_v3/vulkan` (injected).

## MISMATCH — memory / value (serious)
| operator | count | eager → device | file |
|---|---:|---|---|
| `copy` | 116 | input → unrelated buffer | [vulkan-copy-aliasing.md](vulkan-copy-aliasing.md) |
| `full` / `fill` | 42 | `full(1.0)` → 0.0 | [vulkan-full-wrong-constant.md](vulkan-full-wrong-constant.md) |
| `index_put` | 213 | writes misplaced | [vulkan-index_put-misplace.md](vulkan-index_put-misplace.md) |
| `clamp.Tensor` | 308 | in-range values → 0.0 | [vulkan-clamp-zeroes.md](vulkan-clamp-zeroes.md) |

## MISMATCH — non-finite domain
`gelu` (371), `mean` (139→65504), `floor_divide` (430), `sqrt` (40), `rsqrt` (29), `addmm` (24) —
[vulkan-nonfinite-domain.md](vulkan-nonfinite-domain.md).

## CRASH — native abort cluster (~555)
`clone`, `lift_fresh_copy`, `alias_copy`, `max_pool2d_backward`, `unfold_copy`,
`split_with_sizes_copy`, … — [vulkan-crash-cluster.md](vulkan-crash-cluster.md).

Full SKIP/CRASH reason tables: [../skips.md](../skips.md). Ruled out: [../ruled_out/](../ruled_out/).
