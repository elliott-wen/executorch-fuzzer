# Cross-device differential — Ethos-U vs Vulkan vs QNN

Confirmed **single-op** operator bugs per backend (device-verified by single-op isolation):

| backend | device | confirmed op bugs |
|---|---|---:|
| Vulkan | Moto G54 (Android 14) | 49 |
| Vulkan | ASUS a12201 | 24 |
| QNN HTP | x86 emulator (fp16) | 32 |
| **Ethos-U** | **Corstone-300 FVP (int8/Vela)** | **26** |

The comparison below matches op names canonically (dots/underscores normalized). **Caveat:** the
Ethos-U set here is the **26 first-pass isolation candidates**; only **5 survive 5× stability**
(`REVERIFICATION.md`). This "which ops diverge across backends" view is still useful at the candidate
level, but for confirmed Ethos-U bugs use the 5 stable ops. Notably two of the three all-backend ops
(`bitwise_left_shift` 3/5, `rsub` 1/5) are only intermittent on Ethos-U; `prod` is stable.

## Shared across all three backends
- **`bitwise_left_shift.Tensor_out`** — dtype/value divergence on Vulkan (fp16), QNN, and Ethos-U
  (float32 vs uint8). A shift op wrong on three independent backends points at a **shared
  ExecuTorch decomposition / type-promotion** issue, not a per-device kernel.
- **`rsub.Scalar`** — value divergence on all three.
- **`prod`** — reduction divergence on all three (non-finite on Ethos-U).

## Shared Ethos-U ∩ QNN (the two quantized-ish backends)
`_upsample_bilinear2d_aa.out`, `bitwise_right_shift.Tensor_Scalar_out`, `div.Scalar`, `elu.out`,
`pow.Scalar_out` — plus the all-three pair. QNN (fp16) and Ethos-U (int8) share these despite very
different numeric paths, suggesting the fault is in the **lowering/decomposition** common to both,
not the numeric backend.

## Shared Ethos-U ∩ Vulkan
`clamp.Tensor_out`, `fill.Scalar` (+ the all-three pair).

## Ethos-U-unique (16) — int8 / Vela specific
`_adaptive_avg_pool2d`, `addmm`, `alias_copy`, `any.dims_out`, `bitwise_right_shift.Tensor_Scalar`,
`bitwise_right_shift.Tensor_out`, `eq.Tensor_out`, `floor_divide` (`.default` + `.out`), `logit`,
`mean.default`, `mean.out`, `native_group_norm`, `native_layer_norm`, `slice_scatter`. These cluster on **reductions** (`mean`,
`any`), **normalizations** (`group_norm`, `layer_norm`), **scatter/alias** (`slice_scatter`,
`alias_copy`), and **pooling** (`_adaptive_avg_pool2d`) — the ops whose int8 output requant / accumulator
handling is Vela-specific.

## The important structural difference
Bug **counts** are comparable (23 vs 32–49), but the **profile differs sharply**:

| | Vulkan | QNN | **Ethos-U** |
|---|---|---|---|
| dominant failure mode | genuine per-op kernel bugs | buffer-aliasing graph-opt bug + kernels | **graph-context requantization** |
| single-op kernel bugs | 49 (the bulk) | 32 | 23 (the *minority*) |
| graph-level share | small | ~56% of GRAPHOPT is one aliasing bug | **~56% of diverging ops are CLEAN-in-isolation** |

On Vulkan the mismatches *are* the kernels. On Ethos-U the kernels are mostly correct in isolation;
the risk lives in **Vela's graph compilation** (`GRAPH_CONTEXT.md`). QNN sits between: a dominant
graph-opt (buffer-aliasing) bug plus a kernel set. So the same corpus surfaces a **structurally
different bug population per backend**, which is the point of the differential.
