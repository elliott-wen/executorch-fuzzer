# Moto G54 — confirmed Vulkan delegate bugs

Every bug below is **device-confirmed on the Moto G54 5G** with a self-contained repro that
builds the `.pte`, runs it on the phone via the broker, and prints eager vs device. Run any with
`python mobile/findings_v2/vulkan_moto/bugs/repro_<name>.py` (needs a connected Vulkan worker on
`127.0.0.1:15554`).

## Memory-planning / aliasing (the most serious — silent wrong data)
| bug | repro | summary |
|-----|-------|---------|
| [vulkan-copy-elision-aliasing.md](vulkan-copy-elision-aliasing.md) | `repro_vulkan-copy-elision-aliasing.py` | copy elided to a buffer alias; planner frees the still-live source → a **sibling output is silently overwritten** (`out[0]` 0.8486 → 2.4668) |

## Kernel divergences (single-op, isolation-confirmed)
| bug | repro | eager → device |
|-----|-------|----------------|
| [vulkan-floor-divide-self.md](vulkan-floor-divide-self.md) | `repro_vulkan-floor-divide-self.py` | `floor_divide(x,x)`: `1` → `0` (device `div(x,x)`=1.0, so bug is in floor_divide) |
| [vulkan-left-shift-overwidth.md](vulkan-left-shift-overwidth.md) | `repro_vulkan-left-shift-overwidth.py` | `int32 [1,2,3,4]<<40`: `[0,0,0,0]` → `[256,512,768,1024]` (shift by 40 mod 32); right-shift same |
| [vulkan-scatter-src-drops-writes.md](vulkan-scatter-src-drops-writes.md) | `repro_vulkan-scatter-src-drops-writes.py` | `scatter.src_out([1,2,3,4],0,[0,2],[5,6])`: `[5,2,6,4]` → `[0,2,3,4]` (writes dropped, elem 0 →0) |
| [vulkan-index-put-drops-writes.md](vulkan-index-put-drops-writes.md) | `repro_vulkan-index-put-drops-writes.py` | `index_put([1,2,3,4],[[1,3]],[7,8])`: `[1,7,3,8]` → `[0,2,3,4]` (writes dropped, elem 0 →0) |

## Validation rejections (SKIP, isolation-confirmed) — see ../ISOLATION_PROBES.md
- `native_group_norm` — rejected in all weight forms.
- `native_layer_norm` — requires constant, non-None weight and rank-1 normalized_shape.

## Investigated but NOT filed (ruled out by isolation)
`clamp.Tensor_out`, `select_scatter`, `max`, `mean`, `var.correction`, `_softmax`, `sinh`,
`unfold_copy`, `narrow_copy` — all **correct in single-op isolation**. Their corpus mismatches are
the aliasing/memory-planning class, not standalone kernel bugs.
