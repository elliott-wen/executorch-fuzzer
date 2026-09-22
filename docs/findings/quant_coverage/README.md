# Per-operator quantization coverage, per backend

Answers: **how many operators can each backend actually quantize?** Measured, not read off
declarations. Input to the `corpus_v6` decision (single-op corpus with `--quantize` on).

## Method

209 distinct operators, one representative **single-operator** graph each, taken from
`corpus_v5/portable` (`build_oplist.py` → `oplist.tsv`). For each op and backend, `qsweep.py`:

1. exports the graph, then runs the backend's own PT2E quantizer through
   `prepare_pt2e` → calibrate → `convert_pt2e` (`base._convert_pt2e_module`);
2. counts inserted `quantized_decomposed.{quantize,dequantize}_per_{tensor,channel}` nodes.
   **ANNOTATED = the quantizer actually quantizes that op.**
3. for ANNOTATED ops only, runs `build_job(src, backend, quantize=True)` and records
   READY + `delegated_ops`.

ANNOTATED is deliberately independent of whether the backend's quantized *lowering* works, so
backends with a broken quantized path (qualcomm) are still measured. Validated against a
known-good case: `relu(add(L0,L1))` on xnnpack → 6 q/dq nodes.

Reproduce: `python qsweep.py <backend> out.tsv [--limit N] [--no-lower]` with
`MOBILE_BACKENDS=<backend>` set (backends corrupt each other in-process). VGF additionally needs
`MODEL_CONVERTER_LIB_DIR=/opt/nvidia/nsight-systems/2024.6.2/host-linux-x64`; qualcomm needs
`android-dev/android-env.sh` sourced; mediatek needs `.venv-mtk`.

## Results (209 ops)

| backend | QuantMode | ANNOTATED | not | ERR | lowers READY | delegates |
|---|---|--:|--:|--:|--:|--:|
| cortex-m | ALWAYS | **135** | 62 | 12 | — | — |
| samsung | OPTIONAL | **122** | 85 | 2 | — | — |
| nxp | ALWAYS | **122** | 8 | 79 | — | — |
| mediatek | OPTIONAL | **74** | 134 | 1 | **38** | **38** |
| qualcomm | OPTIONAL | **20** | 184 | 5 | **14** | **6** |
| ethos-u | ALWAYS | **14** | 183 | 12 | — | — |
| vgf | OPTIONAL | **14** | 183 | 12 | 13 | **9** |
| xnnpack | OPTIONAL | **1** | 208 | 0 | 0 | 0 |
| vulkan | OPTIONAL | **0** | 208 | 1 | 0 | 0 |
| coreml | OPTIONAL | *sweep running* | | | | |

`out_samsung.tsv` is a partial run (46 ops) that died when the ENN compiler crashed the host
process at `bitwise_xor.Tensor_out` during quantized lowering; `out_samsung_ann.tsv` is the
complete annotation-only re-run. A first, crashed Samsung run reporting "16 of 46" is superseded.

## Findings

**XNNPACK is inert on single-operator graphs: 1 of 209** (`linear`, which then fails to lower with
a `ChannelsLastTagged` pass error). `XNNPACKQuantizer` annotates **fusion patterns**
(conv+relu, linear+activation), not bare operators — `relu(add)` annotates 6 q/dq nodes while the
same ops in isolation annotate none. **Quantized XNNPACK coverage requires `--nodes >= 2`**, which
conflicts with a single-operator corpus design.

**Vulkan is structurally zero: 0 of 209.** `VulkanQuantizer` is weight-only and annotates only
`linear`. Weight quantization needs real **parameters**; the fuzzer emits every input as a random
leaf, so no generated graph has a weight to quantize. `linear` itself ERRORs. Weight-only
quantization can never fire on this corpus — do not generate `corpus_v6/vulkan`.

**Ethos-U and VGF annotate the identical 14 ops** — `abs`, `alias_copy`, `clone`, `detach_copy`,
`expand_copy`, `index_put`, `index_select`, `mean`, `permute_copy`, `pixel_unshuffle`, `repeat`,
`slice_copy.Tensor`, `split_copy.Tensor`, `unfold_copy` — because both use
`executorch.backends.arm.quantizer` with `get_symmetric_quantization_config`. Note the list is
almost entirely **data-movement** ops, not arithmetic: a mechanical explanation for the low
Ethos-U corpus yield already recorded in `findings/ethos_u/`.

**Even int8-only (ALWAYS) backends annotate a minority of ops** — Ethos-U 14/209. On an
integer-only NPU an unannotated op cannot delegate, so this bounds their corpus yield directly.

## Blockers found (each would break a `corpus_v6` fleet)

1. **coreml — FIXED (two bugs).** (a) `coreml.py` set `"activation_dtype": "qint8"`, but
   coremltools validates that field against `[torch.quint8, torch.float32]` only — every graph
   failed at `prepare_pt2e`. Now `quint8` activations / `qint8` weights, the standard pairing for
   the symmetric scheme. (b) That exposed a second failure: the MIL converter has no `quantize` op
   before **iOS17** (`No available version for quantize in the coremltools.target.iOS15 opset`), so
   the quantized path now passes an explicit
   `CoreMLBackend.generate_compile_specs(minimum_deployment_target=ct.target.iOS17)`. The **float
   path keeps the default target**, so existing float corpora stay reproducible.
   Verified: `quantize=False → READY deleg=2` (unchanged), `quantize=True → READY deleg=8`.
2. **qualcomm — FIXED.** Annotated 20 ops but `to_executorch` raised
   `AssertionError: Failed to generate Qnn context binary`: the partitioner reports
   `quantize_per_tensor | True` but `dequantize_per_tensor | False`, so the QDQ cluster was split at
   the partition boundary. The quantized path now routes through
   `to_edge_transform_and_lower_to_qnn`, which carries QNN's QDQ transform passes; float keeps the
   generic `to_edge`+`to_backend` path and its measured ~53% vs ~42% yield advantage.
   **Before: 0 READY / 0 delegating. After: 14 READY / 6 delegating**
   (`abs`, `atan2.out`, `index_put`, `index_select`, `roll`, `squeeze_copy.dims`).
   ⚠️ The sweep **hard-crashed the host process** at `repeat` (no Python exception) — the known
   non-deterministic QNN AOT heap corruption (`findings_v2/qnn_asus_ai2201/bugs/
   aot-nondeterministic-heap-crash.md`), now also reachable on the quantized path. Run the QNN
   fleet with per-worker isolation and expect attrition.
3. **samsung** — the ENN compiler **hard-crashed the host process** during quantized lowering of
   `bitwise_xor.Tensor_out`. A 100k fleet can die mid-generation; use per-worker isolation.
4. **vulkan / xnnpack** — not bugs, but structurally incapable on a single-op corpus (above).

## Recommendation for corpus_v6

Generate: **samsung** (122 ops), **mediatek** (74 annotated; 38/38 that lower also delegate), **vgf**
(14 → 13 → 9, small but working end to end).
Fix first: **qualcomm**, **coreml**. Skip: **vulkan**, **xnnpack**.
