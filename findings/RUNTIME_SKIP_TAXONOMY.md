# Runtime SKIP taxonomy — per backend

A **runtime SKIP** is categorically different from an AOT `lower`/`export` failure: the `.pte`
already lowered successfully (it is in the corpus), was shipped to a device or host executor, and
the **executor refused to run it**. AOT failures never became jobs at all.

Derived from every `SKIP` row in the per-run skip logs: **53,865 records across 16 runs**,
2,225 distinct normalized messages, **99.9% classified** (61 residual).
Scripts: `scratchpad/classify_rt.py`, `scratchpad/rt_skips.tsv`.

Not covered (no machine-readable skip log in the findings dir): `coreml_mac`, `coreml_mac2`
(prose `skips.md` only), `openvino_single`, `vgf` (multi-op run). `nxp_imxrt700` **has a log but
zero SKIPs** — its 5,471 non-OK rows are 3,289 MISMATCH + 2,182 CRASH.

## Categories

| id | category | n | share |
|----|----------|--:|------:|
| **RO-run** | Opaque subprocess-runner exit — exit code only, cause never captured | 17,951 | 33.3% |
| **RO-rt** | Opaque runtime error code — `[0x12] Invalid argument` / `Internal error`, no detail | 14,633 | 27.2% |
| **RD** | Delegate runtime rejection — the delegate accepted the subgraph at AOT, its runtime graph builder refuses | 11,472 | 21.3% |
| **RG** | Kernel validation guard — an authored `Check failed (...)` naming the condition | 5,895 | 10.9% |
| **RM** | Operator not registered in the runtime kernel set | 3,131 | 5.8% |
| **RH** | Harness / client side — **not** the backend | 499 | 0.9% |
| **RL** | Delegate init failure at method load | 223 | 0.4% |
| ? | unclassified | 61 | 0.1% |

**60.5% of runtime SKIPs (RO-run + RO-rt = 32,584) carry no attributable cause.** That is the
headline result of this table, and it is far worse than the AOT side, where the equivalent opaque
share was 7.5%.

## Per-run × category

| run | RO-run | RO-rt | RD | RG | RM | RH | RL | ? | TOTAL |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| samsung_e9965 | | 5771 | | | 442 | | | | 6213 |
| vulkan_moto | | 962 | 3825 | | 442 | 78 | | | 5307 |
| vulkan_samsung | | 962 | 3825 | | 442 | 78 | | | 5307 |
| vulkan_asus | | 960 | 3822 | | 441 | 78 | | | 5301 |
| qnn_emulator | 4915 | | | | | | | | 4915 |
| ethos_u | 4643 | | | | | | | | 4643 |
| cortex_m | 4450 | | | | | | | | 4450 |
| vgf_single | 3943 | | | | | 115 | | | 4058 |
| portable_x64 | | | | 2897 | | | | | 2897 |
| xnnpack_arm64 | | 2279 | | | 429 | 63 | | | 2771 |
| xnnpack_x64 | | | | 2204 | | | 84 | 23 | 2311 |
| qnn_phone | | 1733 | | | 368 | 68 | | | 2169 |
| mtk_phone | | 1366 | | | 442 | 1 | | | 1809 |
| cuda_h200 | | | | 794 | | | 139 | 38 | 971 |
| qnn_samsung_sm8750 | | 341 | | | 70 | 8 | | | 419 |
| qnn_samsung | | 259 | | | 55 | 10 | | | 324 |
| **TOTAL** | **17951** | **14633** | **11472** | **5895** | **3131** | **499** | **223** | **61** | **53865** |

The columns partition almost perfectly by **how the executor is bound**, not by backend:

- **Subprocess runner** (`qnn_emulator`, `ethos_u`, `cortex_m`, `vgf_single`) → 100% RO-run.
- **Android app over JNI** (`samsung_e9965`, `xnnpack_arm64`, `qnn_phone`, `mtk_phone`, `vulkan_*`)
  → RO-rt.
- **In-process host runtime** (`portable_x64`, `xnnpack_x64`, `cuda_h200`) → RG, with the actual
  check text.

## The controlled comparison: diagnosability is a property of the *client*, not the failure

`xnnpack_x64` and `xnnpack_arm64` are the **same backend on the same corpus**, differing only in
executor binding. Their SKIP causes are the same operators — but only one run says why:

| operator | xnnpack_x64 (in-process host) | xnnpack_arm64 (Android/JNI) |
|---|---|---|
| `native_group_norm` | 532 · `[normalization_ops_util.cpp] Check failed (in.size(0) == N)` | 716 · `[0x12] Invalid argument` |
| `arange.start_out` | 288 · `Check failed (utils::extract_scalar(start, &d_start))` | 414 · `[0x12] Invalid argument` |
| `_pdist_forward` | ~340 · `Check failed (tensor_is_rank(in, 2))` | 341 · `[0x12] Invalid argument` |
| `copy` | 167 · `Check failed (non_blocking == false)` | 212 · `[0x12] Invalid argument` |
| `expand_copy` | ~200 · `Check failed (implicit == false)` | 202 · `[0x12] Invalid argument` |

**RO-rt and RG are the same underlying phenomenon.** The kernel raises an identical authored
guard in both runs; the Android path discards the message and surfaces only the ExecuTorch status
code. So 14,633 records (27%) are not *inherently* unattributable — the information exists at the
point of failure and is lost in the client binding. Fixing the JNI layer to propagate the error
string would reclassify essentially all of RO-rt into RG.

The same op-level agreement holds across every Android run: the RO-rt bucket is dominated
everywhere by `native_group_norm`, `arange.start_out`, `_pdist_forward`, `copy`, `expand_copy`,
`scatter*` and the `replication_pad*`/`reflection_pad*` family.

---

## Category detail

### RO-run — opaque subprocess-runner exit (17,951)
`qnn_emulator` 4,915 · `ethos_u` 4,643 · `cortex_m` 4,450 · `vgf_single` 3,943

**All 17,951 are exit code 2.** These backends are driven through a wrapper shell script rather
than an in-process runtime, and every wrapper shares one convention:

```
exit 0 = ran, outputs written
exit 2 = graph not runnable   -> client reports SKIP
exit 3 = environment not set up
other  = hard failure         -> client reports CRASH
```

So a SKIP here means: *the graph failed to load or execute on the target and produced no output
tensors*, and the cause went to a log the harness did not retain (`run.log`, or
`/tmp/fvp_job_XXXX/uart.log`). The observed forms:

- `qnn_runner rc=2:` — 4,915, no text whatsoever.
- `vgf_runner rc=2:` — 3,425 bare, plus **518 that report `Segmentation fault (core dumped)`**.
- `fvp_runner rc=2: >> building runner: … (shared SDK, no fetch) … no out-*.bin produced
  (see /tmp/fvp_job_XXXX/uart.log)` — ethos_u and cortex_m.

**Careful reading required on the FVP ones.** `>> building runner: …` is *progress* output, not a
build failure: the runner built fine, the FVP executed, and emitted no output tensors. An earlier
draft of this taxonomy mis-filed them as harness artifacts on the strength of that build text.

**RO-run is opaque in the aggregate log ONLY — the per-backend analyses resolved it.** All four
runs in this category carry a `skips.md` with the verbatim device cause, recovered at analysis time
by re-running representative jobs against a persistent output directory and reading the
`uart.log` / `run.log` (see `cortex_m/_work/cm_repro.py`). So the 17,951 figure measures what the
*feeder skip-log* preserved, not what is knowable:

| run | where the resolved causes live | dominant signature |
|---|---|---|
| `vgf_single` | `vgf_single/skips.md` | `Failed to process VGF blob → Init failed for VgfBackend 0x1` (converter blob won't JIT-compile at method load); plus `where.self` (108) `Output tensor byte size 4 != IO allocation 2` — a delegate IO dtype-width bug |
| `ethos_u` | `ethos_u/skips.md` | `no out-*.bin produced` on degenerate constant graphs (`ones/zeros/arange/full.out`) and `bitwise_*_shift` failing to execute on the FVP |
| `cortex_m` | `cortex_m/skips.md` | `cortex_m::` kernel guards vs portable-fallback guards, split by which kernel the `.pte` calls (`_work/cm_kernels.py`) |
| `qnn_emulator` | `qnn_emulator/skips.md` | the emulator additionally yields real crash signals — 444 aborts + 165 segfaults with core dumps, unlike the phones' opaque native abort |

The engineering fix is therefore narrower than "capture more": the reason string already reaches
`run.log`/`uart.log`, and the per-backend workflows already read it. What is missing is
**propagating it into the feeder skip-log** so the aggregate is analysable without a re-run.

#### The SKIP/CRASH boundary here is a harness convention, not a property of the failure
The wrappers do not merely forward the runner's exit code — they **grep the log and map matches to
exit 2**. `fvp_runner.sh` maps `Missing operator|kernel '.*' not found|Loading of
method.*failed|assert failed|Method::execute.*0x` to SKIP, i.e. an on-target **assertion failure
is deliberately classified as SKIP**. `vgf_runner.sh` is broader still, its pattern list including
bare `Vulkan|VGF|volk|vkCreate`, which matches almost any Vulkan log.

The measurable consequence: **518 vgf_single records are genuine native crashes
(`Segmentation fault (core dumped)`) recorded as SKIPs**, because the wrapper rewrote the segfault
exit status to 2. The same mechanism can absorb FVP assertion failures.

Two implications for any use of these numbers:

1. **SKIP and CRASH rates are not comparable between subprocess-runner runs and in-process runs.**
   For `qnn_emulator`, `ethos_u`, `cortex_m` and `vgf_single`, the split reflects a grep in a shell
   script; for `portable_x64`, `xnnpack_x64` and `cuda_h200` it reflects whether the process
   actually died.
2. The 518 known-segfault rows should be reported as crashes, and the wrappers' pattern lists
   narrowed (match the runner's own exit code, and treat signal-terminated exits as CRASH
   unconditionally) before the next run.

### RO-rt — opaque runtime error code (14,633)
`[ExecuTorch Error 0x12] Invalid argument: Execution failed for method: forward` (10,291) and
`Internal error: …` (4,339), with no further detail. Exclusively the Android/JNI runs. See the
controlled comparison above — this is RG with the message stripped.

### RD — delegate runtime rejection (11,472)
**Vulkan only**, and near-identical on all three phones (3,825 / 3,825 / 3,822) because it is
device-independent. The partitioner delegated the node; the C++ graph builder then throws:

| n (per device) | check |
|--:|---|
| ~812 | `NativeLayerNorm.cpp` — `native_layer_norm requires weight to be non-None` |
| ~713 | `GroupNorm.cpp` — `VK_CHECK_COND(graph.val_is_tref(weight_data))` (weight must be a prepacked constant) |
| ~445 | `BatchNorm.cpp` — `(in_sizes.size() == 4) is false! BatchNorm only support 4d tensor` |
| ~317 | `ComputeGraph.h` — `Cannot extract scalar from Value with type NONE` |
| ~542 | `VecUtils.h` — `(ints.size() == 2) is false!` / `Index out of bounds!` (avg_pool2d, softmax) |
| ~340 | `ArgReduce.cpp` needs buffer storage / `BlitNode` needs texture storage — contradictory |
| ~195 | `ShaderRegistry.cpp` — `Could not find ShaderInfo with name binary_eq_buffer_int8` |

Root cause: Vulkan's partitioner registry (`op_registry.py` `OpFeatures`) declares only **dtype**
and **storage**. It cannot express argument optionality, constness, rank, or int-list arity, so it
over-accepts and the runtime discovers the mismatch. This is the **runtime instance of the same
partitioner over-acceptance** seen at lowering in `corpus_v5/FAILURE_TAXONOMY.md`.

### RG — kernel validation guard (5,895)
The only category where the log states the condition. `portable_x64` 2,897 · `xnnpack_x64` 2,204 ·
`cuda_h200` 794.

Portable/xnnpack classes: rank/shape/dtype guards (`in.size(0) == N`, `tensor_is_rank(in, 2)`,
`t.dim() == rank`), scalar-extract guards (`extract_scalar`), int64-index requirements
(`index.scalar_type() == Long`), explicitly unimplemented forms (`non_blocking == false`,
`implicit == false`), dim-order (`all_contiguous || all_channels_last`), and unhandled dtypes
(`Unhandled dtype Float8_e4m3fn for permute_copy.out`).

`cuda_h200` is a distinct sub-class — all 794 are **delegate/runtime tensor-contract violations**
at `CALL_DELEGATE`, not kernel arg checks: `SlimTensor storage_offset must be 0, got N` (389),
`dtype mismatch, SlimTensor dtype N != ETensor dtype N` (213), `SlimTensor must be contiguous`
(192).

### RM — operator not registered (3,131)
**100% `_fft_r2c`**, at 441–442 per device run (429 xnnpack_arm64, 368 qnn_phone, 70/55 on the
short Samsung runs). The op is not delegated and has no entry in
`kernels/portable/functions.yaml`, so the on-device kernel registry cannot resolve it — a whole-op
loss on every device.

Note it is **absent from the host runs**: `portable_x64`, `xnnpack_x64` and `cuda_h200` reach a
real `_fft_r2c` kernel (and fail later on `onesided=False` or dim-order instead), because the
installed ExecuTorch package registers a broader kernel set than the Android app builds. So
"portable kernel" is not one fixed set — it depends on the runtime build.

### RL — delegate init failure at load (223)
Fails before execution, at `Init` / method load:

- `cuda_h200` 139 — `Failed loading symbol AOTInductorModelContainerCreateWithDevice` (99) and
  `Init failed for backend CudaBackend` (40). The AOTI `.so` in the `.pte` cannot be loaded.
- `xnnpack_x64` 84 — `[XNNCompiler.cpp] Failed to define tensor N with code:
  xnn_status_unsupported_parameter`. XNNPACK accepted the partition at AOT and rejects the tensor
  at subgraph construction — the same over-acceptance shape as Vulkan's RD, just caught at init.

### RH — harness / client side (499) — EXCLUDE FROM BACKEND CLAIMS
`unsupported input dtype code N on Android` (327, our Java client refuses before the runtime),
`output decode ValueError: both buffer length and count must not be 0` (115, our client could not
rebuild the returned tensors), `ArrayIndexOutOfBoundsException: vector` (57).

### ? — unclassified (61)
`cuda_h200` 38 (a `/tmp/torchinductor_*` path in the message), `xnnpack_x64` 23
(`[XNNExecutor.cpp] Internal Error: Propagating input shapes failed`).

---

## Reading guidance

1. **Do not compare SKIP rates across runs without controlling for executor binding.** A
   subprocess-runner run reports 100% opaque; an in-process host run reports the actual check for
   the identical failure. Worse, for subprocess runners the SKIP/CRASH split is decided by a grep
   in the wrapper script, which demonstrably swallows real crashes (518 confirmed segfaults inside
   RO-run) — see that section.
2. **RD and RL are the runtime face of partitioner over-acceptance** (Vulkan 11,472, XNNPACK 84).
   Combined with the AOT figure, over-acceptance is the single most cross-cutting defect class in
   the study, and the stage at which it is caught is incidental.
3. **RM is a runtime-build property, not a backend property** — the same operator resolves on the
   host and not on the phone.
4. **The actionable engineering item is diagnostic plumbing, not backend coverage.** Propagating
   the kernel error string through the JNI client (RO-rt, 14,633) and capturing the runner's
   `uart.log`/stderr (RO-run, 17,951) would move 60% of all runtime SKIPs from unattributable to
   attributable without changing a single backend.
