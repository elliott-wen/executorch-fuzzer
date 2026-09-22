# Vulkan SKIP census — 72,285 graceful refusals (specific reasons)

A SKIP is the Vulkan delegate / ET runtime declining a graph *gracefully* (a caught exception →
SKIP, not a crash). Generators: `tmp/run_vulkan_galaxy_26/skip_aggregate.py` (coarse census,
`skip_census.txt`) and `tmp/run_vulkan_galaxy_26/skip_specific.py` (the per-category specific
breakdown below, `skip_specific.txt`).

Two surfaces: ~40.7k carry a full C++ frame (`File.cpp:line` + predicate + message) → decoded
precisely; ~31.5k come back as **Java/runtime exceptions that strip the C++ frame** (worker
limitation) → only op-correlated (see last section).

## Decoded C++-frame refusals (~40.7k)

### missing-shader — 13,166  (asymmetric dtype coverage in shader generation)
| n | specific reason |
|---:|---|
| 10,965 | **no `*→uint8` convert shader** — `view_convert_buffer_{int32,float,half}_uint8`. The convert matrix in [view_convert_buffer.yaml](../../../pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/glsl/view_convert_buffer.yaml) lists `uint8` only as a **source** (`uint8→float/half/int32`), never a **destination**. So any graph whose output is **bool / uint8** (every comparison/logical op result) has no copy-out conversion → SKIP. |
| 1,375 | **int32 equality variant misnamed** — lookup wants `binary_eq_buffer_int32` but the yaml hand-named the int32 variant `binary_eq_int32_buffer` (dtype in the middle) → name miss. [binary_op_buffer.yaml](../../../pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/glsl/binary_op_buffer.yaml) |
| 826 | **int32 absent from `index_tensor` dtype set** — `index_tensor_{buffer,texture3d}_int32` not generated (int32-indexed gather/index). |

These three are **fixable shader-gen gaps**, not inherent limits: add the `→uint8` combos, fix the
`binary_eq` int32 name, add int32 to `index_tensor`. They alone are ~13k of the 72k.

### broadcast / out-shape — 7,256
`BinaryOp.cpp:35 (out_sizes == broadcasted_sizes)` — the binary op's pre-sized `out=` buffer shape
doesn't equal the broadcasted input shape. Corpus emits out-variant binary ops whose `out=` tensor
was allocated at a different (un-broadcasted) shape.

### storage-layout — 6,115
- 3,974 `BlitNode.cpp:32` — blit/copy supports **only texture-backed** tensors, got a buffer.
- 2,141 `ArgReduce.cpp:28` — `argmax`/`argmin` requires **buffer** storage, got a texture.

(The delegate disagrees with itself on which storage each op wants → many copies refuse.)

### reduce-dim-unsupported — 5,421
- 5,250 `Reduce.cpp` — **"Only 1 or 2 dimensions supported"** for `mean`/`sum`/`amax`/`amin` (can't reduce ≥3 dims at once).
- 151 `ArgReduce.cpp:37` — `argmax`/`argmin` only along the **last** dim.
- 20 — softmax/reduce where reduce dim == packed/concat dim.

### rank-unsupported — 4,744
- 1,853 `BatchNorm.cpp:65` — batch_norm **4-D only**.
- 1,846 `VecUtils.h:275 operator[]` — vec index OOB (dim/rank beyond support).
- 995 `VecUtils.h:367 make_ivec2` — expects **exactly 2** spatial dims (pool/conv params).
- 50 `Tensor.h` — tensor rank > 4 unsupported.

### weight-must-be-constant — 4,042
norm/conv params must be a constant `TensorRef`, but the corpus feeds a **computed** tensor:
2,760 `group_norm` (GroupNorm.cpp:188), 767 `Value.h:266`, 430 `Staging.cpp:217` (prepack), 85
`batch_norm` (BatchNorm.cpp:44).

### dynamic-scalar — 3,823
`ComputeGraph.h:596 extract_scalar` — a scalar argument must be **static/const** at build time.

### conv-arg / staging-size — 59
47 `conv1d` only `groups==1`; 12 staging buffer too small.

## Opaque refusals (~31.5k) — C++ frame stripped by the worker

These return as Java/runtime exceptions with **no kernel detail**, so they can only be op-correlated.
Notably **`to` (`_to_copy`/dtype conversion) dominates every bucket** — strong evidence these are the
same dtype-conversion + int64/uint8 failures as above, just surfaced through a path that loses the
C++ `what()` string.

| bucket | n | surfaced as | dominant ops (hint) |
|---|---:|---|---|
| other-runtime | 12,292 | misc Java exception | `to`, `slice`, `bcast`, `div`, `native_layer_norm`, `mean` |
| delegate-build-oob | 11,031 | `ArrayIndexOutOfBoundsException: vector` (host graph build) | `to`, `bcast`, `mean`, `min`, `max` |
| runtime-exec-failed (0x12) | 7,298 | `[ExecuTorch Error 0x12] Invalid argument` | `to`, `slice`, `bcast`, `remainder` |
| op-not-implemented (0x14) | 861 | `[ExecuTorch Error 0x14] Operator missing` | `to`, `slice`, `bcast`, `fmod`, `pow` |

**To make these specific:** fix the Android worker to propagate the C++ exception `what()` string
(the kernel `File.cpp:line` + predicate) instead of collapsing to a Java exception / bare error code
— same idea as the existing `jni_layer` exception→SKIP path. That would move most of this 31.5k into
the decoded buckets above.

## Reading

The C++-frame skips are **coverage gaps, not bugs** — they define the effective Vulkan op surface on
this corpus (~28% of graphs run at all). The single biggest lever is the **shader-gen dtype matrix**:
adding `→uint8`, fixing `binary_eq` int32, and adding int32 `index_tensor` removes ~13k refusals and
likely a chunk of the `to`-dominated opaque 31.5k too. The `VecUtils.h:275/367` bounds *throws* are
the graceful siblings of the unguarded bounds violations that abort as CRASHes ([../crash/README.md](../crash/README.md)).
