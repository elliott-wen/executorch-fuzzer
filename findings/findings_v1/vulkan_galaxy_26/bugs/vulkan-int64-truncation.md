# int64 silently truncated to int32 on the Vulkan/Adreno GPU

**Signature:** root ops `_to_copy` / `min` / `max` / `bitwise_left_shift` / `bitwise_right_shift` carrying `torch.int64` — `kind: delta` (wrong integer magnitude)
**Cluster size:** ~278 born-here from `_to_copy` alone (the single biggest born-here root cluster), plus the int64-reduction/shift carriers below. Flagged-op carriers seen: `unfold_copy` (`w33:241`, `w52:1661`, `w73:383`), `min` (`w1:1582`), `rsqrt` (`w78:369`), and any consumer of an int64 producer.
**Classification:** `int64-limitation` (the Adreno GPU has no 64-bit int; backend downcasts to int32 by design). Two sub-cases are arguably a **real-bug** quality issue: the downcast is a *silent* low-32-bit truncation with no saturation/guard, so values that overflow int32 wrap to nonsense, and one input graph (w78) feeds a negative shift amount which is C/GLSL UB.

## What happens

Eager runs int64 in true 64-bit; the device stores int64 as int32 and converts on the host copy via `static_cast<int32_t>` (low-32-bit truncation), so the device value is the low 32 bits of the eager int64 reinterpreted as int32 (then sign-extended back to int64 on read-out).

| job | root op | eager | vulkan (device) | why |
|-----|---------|-------|-----------------|-----|
| `w33:241` | `_to_copy(nan→int64)` | `-9.223e18` (INT64_MIN, `0x8000000000000000`) | `0` | low 32 bits of INT64_MIN are `0x00000000` |
| `w1:1582` | `min` over **empty** int64 (identity = INT64_MAX `0x7FFFFFFFFFFFFFFF`) | `9.223e18` | `-1` | low 32 bits `0xFFFFFFFF` = int32 `-1` |
| `w78:369` | `bitwise_left_shift(1, -3)` | `0` | `2.306e18` (`1<<61`) | negative shift is UB; device computed `1 << ((-3)&63)` = `1<<61` |

All three eager values were reproduced on host (CPU) — see repro.

## Root cause

The int64-has-no-GPU-representation decision and the truncation are in two files:

1. **Where int64 is mapped to int32 storage** —
   `pytorch_ref/executorch/backends/vulkan/serialization/vulkan_graph_builder.py:218-239`.
   `downcast_64_bit` defaults to **True** (`vulkan_graph_builder.py:50`). `get_effective_dtype` rewrites the on-GPU tensor dtype:
   ```python
   elif self.downcast_64_bit and dtype == torch.int64:
       return torch.int32
   ```
   So every int64 GPU tensor is physically an int32 buffer/image; all arithmetic (shifts, reductions, casts) runs in 32-bit on the shader.

2. **Where the actual bit-truncation happens** — the host-side staging copy:
   `pytorch_ref/executorch/backends/vulkan/runtime/graph/ComputeGraph.cpp:952-983` (`maybe_cast_and_copy_into_staging`) and `:995-1027` (`maybe_cast_and_copy_from_staging`):
   ```cpp
   if (src_data_dtype == vkapi::kLong && staging_dtype == vkapi::kInt) {
     const int64_t* casted_data = reinterpret_cast<const int64_t*>(data);
     staging->cast_and_copy_from<int64_t, int32_t>(casted_data, numel);   // <-- truncation
   }
   ```
   `cast_and_copy_from` / `cast_and_copy_to` in
   `pytorch_ref/executorch/backends/vulkan/runtime/api/containers/StagingBuffer.h:84-96, 114-126` is a plain element-wise narrowing:
   ```cpp
   for (size_t i = 0; i < numel; ++i)
     dst[i] = static_cast<DST_T>(src[i]);   // int64 -> int32, keeps low 32 bits, no saturation/guard
   ```
   `static_cast<int32_t>(INT64_MIN) == 0` and `static_cast<int32_t>(INT64_MAX) == -1` (verified in C), which exactly reproduces the w33 and w1 device values. The read-back path (`cast_and_copy_to<int32_t,int64_t>`) sign-extends int32 `-1` back to int64 `-1`, giving the localizer `b[0]=-1`.

Note the comment at `vulkan_graph_builder.py:228-233`: the 32↔64 conversion is *intentionally* done on the CPU staging copy (not in a shader), which is precisely the truncation site above.

Two source-graph artifacts feed into this:
- **w33/w52/w73**: `rsqrt` of a negative gives `nan`; `nan→int64` is INT64_MIN in eager (and `int(nan)` is itself GPU-undefined). Both representations are "garbage", but they disagree.
- **w78**: `bitwise_left_shift(1, -3)` — a **negative shift** is undefined in C and GLSL. Eager returns 0; the GPU evidently masks the shift to the type width (`(-3) & 63 = 61`, `1<<61 = 2.306e18`). Even ignoring the int64→int32 issue this is an unguarded UB.

## Minimal repro

`repro_vulkan-int64-truncation.py` — builds the three tiny cases, runs eager on host, and prints the int32-truncation that the Vulkan staging copy performs (`static_cast<int32_t>`), showing it equals the device values `0`, `-1`, `2.306e18`. References the file:line of the truncation in comments. No device needed.

## Notes

Vulkan-specific (Adreno has no 64-bit int storage/shader support; the backend's
`downcast_64_bit=True` is the deliberate mitigation). The `nan→int` half of the
`_to_copy` cluster overlaps in spirit with portable/xnnpack nonfinite findings,
but the *truncation* mechanism is unique to Vulkan's int32 staging. The honest
classification is `int64-limitation`, but the *silent* low-32-bit wrap (no
saturation, no warning) and the unguarded negative-shift are quality bugs a more
careful impl could clamp/diagnose rather than emit a wrong finite integer.
