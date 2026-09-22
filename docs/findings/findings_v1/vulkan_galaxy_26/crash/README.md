# vulkan_galaxy_26 — CRASH verdicts (6,214 native aborts)

Characterization of the **6,214 CRASH** verdicts from running `corpus/vulkan` (100,032
auto-generated graphs) on a real Samsung Galaxy phone (Adreno GPU, ExecuTorch **Vulkan**
delegate). CRASH = the on-device executor process died from a native abort.

> **UPDATE — op-level attribution now exists.** The original "no device signature recoverable"
> conclusion (below) is superseded for *which op*: the [bisect localizer](../bisect-localizer.md)
> prefix-bisects each crash on device and names the culprit op with no logcat. A 150-job sample
> shows **~78% of crashes are the view/copy/reshape/cast family** (`unfold_copy` 21%, `narrow_copy`
> 19%, `alias_copy`, `transpose_copy`, `split_with_sizes_copy`, `lift_fresh_copy`, `clone`,
> `_to_copy`) — real attribution, confirming the enrichment below and tying crashes to the same
> copy/staging machinery as the missing-shader SKIPs and the copy-elision aliasing bug. The literal
> abort *line* still needs `adb logcat`.

## CONSTRAINT (read first): no per-crash device signature exists

A native abort kills the phone executor process outright. The host broker only **INFERS**
CRASH from the dropped binding — see
`android_client/app/src/main/java/com/fuzzer/etclient/ExecutorService.kt:68-69`. There is **no
abort message, no stack, no errno** captured over the wire, and there is **no adb path** to
the remote phone. Every CRASH row carries the identical placeholder reason
`executor process died (native abort)` (verified: 6,214/6,214 in
`tmp/run_vulkan_galaxy_26/skip_reasons_mobile.tsv`).

Therefore root-cause below is **best-effort, derived from**: (a) op-enrichment vs the
full-corpus baseline, (b) re-executing crash graphs in eager on host to recover the true
intermediate shapes/dtypes, and (c) reading the Vulkan + portable source for the enriched
ops. Nothing was run on the device/broker.

## The structural root cause: ET_CHECK aborts, VK_CHECK throws

This is the key mechanism that distinguishes a CRASH from a SKIP. Two different assert
families guard the two execution paths, and they behave differently on failure:

| macro | path | on failure | verdict |
|---|---|---|---|
| `VK_CHECK_COND` / `VK_THROW` | **Vulkan delegate** (graph build + most exec guards) | `throw vkapi::Error` — caught by the executor | **SKIP** (CppException, graceful) |
| `ET_CHECK` / `ET_CHECK_MSG` | runtime / containers | `runtime_abort()` — **kills the process** | **CRASH** |
| `ET_KERNEL_CHECK` | portable CPU kernels | `context.fail()` + `return` — graceful | runtime-exec-failed (SKIP) |

Sources:
- `pytorch_ref/executorch/backends/vulkan/runtime/vk_api/Exception.h:31-44` — `VK_CHECK_COND`/`VK_THROW` **throw**.
- `pytorch_ref/executorch/runtime/platform/assert.h:36-56` — `ET_CHECK[_MSG]` calls `runtime_abort()` → **native abort**.
- `pytorch_ref/executorch/runtime/core/exec_aten/util/tensor_util.h:382-390` — `ET_KERNEL_CHECK_MSG` calls `context.fail()` and returns (graceful).

Consequence: the SKIP census (`tmp/run_vulkan_galaxy_26/skip_census.txt`) is dominated by
the *guarded* Vulkan-build throws — e.g. `VecUtils.h:275 operator[]` "Index out of bounds!"
(1,846) and `VecUtils.h:367 make_ivec2` "vec dim out of range" (995), both `VK_CHECK_COND`.
The **CRASHES are the unguarded siblings of these same bounds/shape violations**: failures
that bypass every `VK_CHECK_COND` and instead hit one of two uncatchable abort vectors:

1. **GPU/driver execution fault (uncatchable).** A degenerate shape that passes graph-build
   validation but produces an out-of-bounds shader write, a bad descriptor, or an invalid
   dispatch. The Adreno driver raises a device fault → `VK_ERROR_DEVICE_LOST` / SIGABRT in
   the worker. There is no C++ exception to catch, so the process dies = CRASH. (Note the
   zero-workgroup case *is* guarded at `DispatchNode.cpp:49-50`, so pure 0-numel delegated
   ops are skipped, not crashed — the crash needs a non-empty-but-malformed dispatch.)
2. **`ET_CHECK` abort in the runtime/containers or at op-resolution** (kernel-not-found for
   an unhandled dtype, e.g. the int64 "no operator implementation" path), which aborts
   rather than throwing.

## What the crash graphs actually contain (host-eager re-execution)

Every crash graph runs cleanly in eager (0/400 sampled threw in eager). Re-executing a
random 400-graph crash sample and inspecting the true output tensors:

| property of the graph (measured in eager) | fraction of crash sample |
|---|---|
| produces an **int64** output (GPU has no native 64-bit int) | **83%** (333/400) |
| produces a tensor with **rank > 4** (5D+; Vulkan textures are 4-dim) | **68%** (273/400) |
| produces a **zero-numel** tensor (degenerate / empty slice) | **57%** (229/400) |
| uses **negative padding** (crop via pad) | **14%** (54/400) |

These overlap heavily — the typical crash graph is a degenerate-shape / unsupported-dtype
combination that eager tolerates but the Vulkan delegate cannot. The crashes are
**predominantly generator-artifacts (degenerate shapes) and int64-limitations**, not
clean wrong-value bugs.

## Op-enrichment table (CRASH presence vs full-corpus baseline)

From `tmp/run_vulkan_galaxy_26/propensity.txt` (n_fail=6,214, n_total=100,032). "Delegated?"
= does the op have a Vulkan kernel in `op_registry.py` / `runtime/graph/ops/impl/`.

| op | in_fail | p_fail | p_base | lift | delegated to Vulkan? |
|---|---|---|---|---|---|
| `unfold_copy` | 1880 | 0.303 | 0.103 | **2.93** | NO → portable CPU |
| `replication_pad3d` | 289 | 0.047 | 0.018 | **2.58** | NO → portable CPU |
| `split_with_sizes_copy` | 676 | 0.109 | 0.051 | **2.14** | YES (`Split.cpp`) |
| `clone` | 1092 | 0.176 | 0.101 | 1.75 | YES |
| `pixel_unshuffle` | 591 | 0.095 | 0.056 | **1.70** | NO → portable CPU |
| `narrow_copy` | 2463 | 0.396 | 0.234 | **1.70** | NO (decomposes; not registered) |
| `tril` | 403 | 0.065 | 0.038 | **1.69** | NO → portable CPU |
| `transpose_copy` | 1005 | 0.162 | 0.103 | 1.57 | NO (only `permute_copy` registered) |

Verified via `grep` over `pytorch_ref/executorch/backends/vulkan/op_registry.py`:
`unfold_copy`, `replication_pad3d`, `pixel_unshuffle`, `tril`, `narrow_copy`,
`transpose_copy` all return **0 hits** (not delegated); `slice_copy` and
`split_with_sizes_copy` return 2 hits each (delegated).

**Why non-delegated ops are crash-enriched:** an op the partitioner rejects runs on the
portable CPU kernel, but it forces a **delegate/CPU boundary** in the middle of the graph.
The Vulkan-delegated *neighbours* of these ops must accept/produce the boundary tensors
(often int64 or 5D), and it is the delegate side of that boundary that faults. So these ops
are *markers* of crash-prone graphs more than the literal abort site.

## Per-top-op root-cause hypotheses

### `unfold_copy` (1880 crashes, lift 2.93) — int64-limitation + generator-artifact
- Not delegated; runs on `pytorch_ref/executorch/kernels/portable/cpu/op_unfold_copy.cpp`.
  Its arg checks (`check_unfold_copy_args`, `copy_ops_util.cpp:986`) are `ET_KERNEL_CHECK` =
  graceful, so the unfold kernel itself does not abort.
- Localizer corroboration: `localize_aggregate.txt` shows `unfold_copy (n=30)`, and example
  rows in `localized.jsonl` show int64/dtype divergence (`eager torch.bool vs backend
  torch.int64`). The brief's "no operator implementation" note points at the **int64
  resolution path**: a delegated consumer of the unfold output has no int64 kernel → hard
  abort at op resolution.
- Example `w0:104` (`corpus/vulkan/w0/w0_104.py`): `n1=min.unary_out` is **0-D scalar**,
  then `n3=unfold_copy(n1, dim=0, size=0, step=6)` → eager shape `(0,)`; downstream
  `narrow_copy(n3,0,0,0)`, `cos.out(out=(0,))`, `ge(out=(0,))` are all zero-numel int64/fp.
  Classic degenerate-shape graph.
- Example jobs: `w0:104 w0:1097 w0:1095 w0:1531 w0:1893 w0:1891`.
- **Classification: int64-limitation (primary) + generator-artifact (0-size).**

### `replication_pad3d` (289 crashes, lift 2.58) — generator-artifact (5D + negative pad)
- Not delegated; portable `op_replication_pad3d.cpp` (checks at lines 25,32 are
  `ET_KERNEL_CHECK` = graceful). pad3d operates on **5D** NCDHW tensors, which Vulkan
  textures cannot represent (`Tensor.cpp:432` switches >4D to buffer storage; many delegated
  consumers then have no buffer-storage shader → fault at the boundary).
- Example `w0:1291`: `replication_pad3d.out(n1, [4,-4,-5,3,2,-5], ...)` — **negative
  padding** (cropping) on a 5D tensor; a sibling output `n5` is shape `(0,0,0,0,0)`.
- Example jobs: `w0:1291 w0:1884 w0:200 w0:331 w0:728 w10:523`.
- **Classification: generator-artifact (5D rank + negative/degenerate pad).**

### `split_with_sizes_copy` (676 crashes, lift 2.14) — generator-artifact (delegated op, degenerate split)
- **Delegated** — `runtime/graph/ops/impl/Split.cpp`. Build-time guard
  `VK_CHECK_COND(out_list->size() == split_sizes.size())` (`Split.cpp:73`) is a throw =
  SKIP. The crash path is the per-split offset loop `Split.cpp:36-39`
  (`split_offset += split_sizes[i]`, raw `std::vector::operator[]`, no bounds check) and the
  per-split sub-dispatch: a zero-size or out-of-range split entry produces a malformed
  copy region → GPU OOB read/write at execution = uncatchable fault.
- Example `w0:1026`: `split_with_sizes_copy(n1,...)` on a graph whose other outputs are 5D
  `(2,2,4,3,2)`; the split feeds `maximum.out` at the delegate boundary.
- Example jobs: `w0:1026 w0:1248 w1:1044 w1:1176 w1:1441 w1:1486`.
- **Classification: generator-artifact (degenerate split sizes) — candidate real-bug** if a
  valid-but-uneven split is shown to OOB; needs a device signature to confirm.

### `narrow_copy` (2463 crashes, lift 1.70) — generator-artifact (highest absolute count)
- Not directly registered (decomposes toward `slice_copy`, which IS delegated via
  `Slice.cpp`). `Slice.cpp` build-side resize uses `in_sizes.at(dim)` and `normalize()`
  (`Slice.cpp:55-66`); a `length=0` narrow yields a zero-extent output. The dispatch path
  faults when the slice region is non-empty-but-out-of-range (the `numel==0` case is caught
  at `DispatchNode.cpp:49`, but a malformed start/step that lands a non-zero workgroup over
  an out-of-range region is not).
- Present in 40% of all crash graphs — it is the most common *marker* op, frequently
  co-occurring with `unfold_copy` (e.g. `w0:104`, `w0:1531`) and 5D tensors.
- Example jobs: `w0:104 w0:1180 w0:1258 w0:1291 w0:1531 w0:1625`.
- **Classification: generator-artifact (degenerate/zero-length narrow).**

### `pixel_unshuffle` (591, 1.70), `tril` (403, 1.69), `transpose_copy` (1005, 1.57)
- All **not delegated** (`pixel_unshuffle`, `tril` → portable CPU; `transpose_copy` → only
  `permute_copy` is registered in Vulkan, `Tensor.cpp:1455` further forbids transposing the
  batch dim of a texture-backed tensor via `VK_CHECK_COND` = SKIP). Like the others, they
  are crash-enriched as **boundary markers**: they force CPU/delegate splits and tend to
  appear in 5D/int64 graphs.
- Example jobs — pixel_unshuffle: `w0:1109 w0:1477 w0:326 w0:383 w0:645`; tril:
  `w0:1105 w0:415 w0:502 w0:53 w0:645`; transpose_copy: `w0:1080 w0:1109 w0:1477 w0:1995`.
- **Classification: generator-artifact (marker ops) + int64/5D-limitation.**

## How to get true crash signatures (if adb to the phone becomes available)

The verdict-level taxonomy above is the ceiling of what is recoverable without the device.
To get *real* per-crash signatures:

1. Get an adb shell on the Galaxy (USB or `adb connect <ip>` if dev-mode wireless is on).
2. `adb logcat -c` then `adb logcat *:E | grep -iE "executorch|vulkan|libc|DEBUG|SIGABRT|VK_ERROR"`
   in one terminal.
3. **Targeted re-feed**: feed ONLY a small batch of the example crash jobs above (one op
   family at a time) through the broker so the abort is isolated in the log window.
4. The abort line will show either:
   - an `ET_CHECK` failure with `file:line` + condition (the runtime/container abort vector), or
   - a Vulkan/driver fault (`VK_ERROR_DEVICE_LOST`, GPU page fault, `vkQueueSubmit` failure)
     pointing at the shader dispatch (the uncatchable vector).
5. Cross-reference the `file:line` with this doc to confirm which of the two abort vectors
   fired per op, and split the "candidate real-bug" classifications (e.g.
   `split_with_sizes_copy`) into confirmed bug vs degenerate input.
6. Optionally rebuild the on-device executor with `-DET_LOG_ENABLED=1` and a tombstone/
   `debuggerd` dump enabled to capture the native stack of the abort.
