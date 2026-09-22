# Quantized vs float, same graphs, same runtime — VGF / Qualcomm / OpenVINO

A **paired A/B** of the three `QuantMode.OPTIONAL` backends that execute locally on this host.
For each backend a single-operator corpus (`--nodes 1`) was generated **twice from the identical
seed schedule**, differing only in `--quantize`, then both were executed on the same local
runtime and diffed job-by-job. Because the seed schedule is a pure function of
`(producer_id, seed, index)`, job `w7:412` is the *same generated graph* in both corpora, so
every comparison below is a graph against itself.

Corpora: `corpus_v7/{vgf,openvino,qnn}_{float,int8}`, paired symlink views `*_p`.
Scripts: `scripts/`. Raw verdict logs: `<backend>/{float,int8}.skip.tsv`.

## 0. What had to be fixed first (three real bugs in our own tooling)

1. **`openvino` could not be quantized at all.** `gen/export/backends/openvino.py` declared
   `supports_quantization = True` as a *class attribute*, shadowing the base **property**, while
   leaving `quant = QuantMode.NEVER`. `build_job`'s guard reads `supports_quantization` and let the
   job through; `base.lower` then raised `RuntimeError: openvino backend does not support
   quantization` on **every** job. `QUANTIZE=1 run_pregen_openvino.sh` was dead on arrival.
   Fixed to `quant = QuantMode.OPTIONAL` (verified: `quantize=True → READY, delegated_ops=6`).

2. **`vgf_runner_io` could not load any quantized `.pte`.** It links `portable_ops_lib` but not
   `quantized_ops_lib`, so every graph whose q/dq nodes stayed *outside* the delegate died at
   method load with `Missing operator: quantized_decomposed::quantize_per_tensor.out` (0x14).
   The runner maps that to **SKIP** — indistinguishable from a genuine backend coverage gap.
   Before the fix the int8 run showed 310 SKIPs and a −14/−12/−12 OK collapse on `detach_copy`,
   `clone`, `view_copy`; after linking `quantized_ops_lib` those all disappear. **Any int8 result
   collected with the old runner is an artifact.**

3. **The Python ET runtime has the same hole.** The pip `_portable_lib` ships without the
   `quantized_decomposed` kernels, so `openvino_client` (which runs in-process) would have had the
   identical false-SKIP problem. Fixed by dlopening the wheel's `libquantized_ops_aot_lib.so`
   `RTLD_GLOBAL` in the executor child (`openvino_client/openvino_client.py`).

## 1. Which operators can each backend actually quantize?

Measured two ways — the second is the honest one.

**(a) One representative graph per op** (209 ops, `findings/quant_coverage/qsweep.py`): run the
backend's own PT2E quantizer `prepare → calibrate → convert`, count inserted q/dq nodes.

| backend | quantizer | ANNOTATED | lowers READY | delegates |
|---|---|--:|--:|--:|
| qualcomm | `QnnQuantizer` | 20 | 14 | 6 |
| openvino | `OpenVINOQuantizer` (nncf MinMax, INT8_SYM) | 17 | 13 | 13 |
| vgf | `VgfQuantizer` (TOSA INT) | 14 | 13 | 9 |

**(b) 3 random real corpus graphs per op** (214 ops, `scripts/ann_rate.py`) — annotation depends on
dtype and shape, so one representative graph *understates* coverage by 40–75%:

| backend | annotated ≥1/3 | always 3/3 | sometimes | never |
|---|--:|--:|--:|--:|
| qualcomm | **35** | 9 | 26 | 178 |
| vgf | **25** | 4 | 21 | 189 |

The three op sets are near-disjoint in character, which drives everything below:

- **openvino** — the matmul family (`linear`, `mm`, `bmm`, `addmm`), elementwise `mul`/`div`/`pow`,
  reductions (`mean`, `sum`), `batch_norm`, `round`. Classic nncf "ops with weights + quantize-agnostic".
- **vgf** — almost purely **data movement**: `clone`, `alias_copy`, `detach_copy`, `expand_copy`,
  `permute_copy`, `slice_copy`, `split_copy`, `unfold_copy`, `repeat`, `index_select`, `index_put`,
  `pixel_unshuffle`, `mean`, `abs`. Barely any arithmetic.
- **qualcomm** — the mixed bag: `convolution`, `linear`, `batch_norm`, `layer_norm`, `atan2`,
  `roll`, `index_put`, `where`, `var_mean`, `select_scatter`, `slice_scatter`.

Quantizer robustness bugs found while sweeping:

- **openvino / nncf, 12 ERRs**: `TypeError: 'NoneType' object is not iterable` on every
  **multi-output** op (`native_layer_norm`, `native_group_norm`, `var_mean`, `split_copy`,
  `unbind_copy`, `_native_batch_norm_legit.no_stats`), and
  `NotImplementedError: "histogram_cpu" not implemented for 'Int'/'Long'/'Byte'` — the
  HistogramObserver is handed integer tensors.
- **qualcomm**: `max_pool2d_with_indices_backward.grad_input` **hard-crashes the host process**
  (`double free or corruption`, no Python exception) during `convert_pt2e`. The sweep needs a
  crash-restart loop (`scripts/ann_qnn_loop.sh`) to get past it.

## 2. What `--quantize` does to corpus generation

Paired on the seed slots both fleets attempted (`scripts/ab_yield.py`):

| backend | attempted by both | READY float | READY int8 | Δ | lost at lowering | gained |
|---|--:|--:|--:|--:|--:|--:|
| vgf | 5,522 | 81.5% | 74.8% | **−6.7 pts** | 423 | 64 |
| openvino | 4,370 | 89.5% | 83.6% | **−5.9 pts** | 271 | 14 |
| qualcomm | 7,109 | 85.7% | 80.6% | **−5.1 pts** | 551 | **190** |

Every backend loses ~5–7 points of corpus yield, essentially all of it at the **lowering** stage.
Qualcomm is the one that meaningfully *gains* graphs too (190) — its QDQ path lowers things the
float path rejects. Top quantized-path lowering failures:

- **vgf** — `_ExportedProgramGraphPassAdapter` pass error (250), `InsertIntNCastsAfterIntNPlaceholdersPass` (190),
  `ReplaceScalarWithTensorByProfilePass` (56), `Expected all tensors to have dtype DType.FPN` on
  `logical_not`/`logical_and` (74).
- **openvino** — `to_executorch TypeError: 'NoneType' object is not iterable` (205, the multi-output
  nncf bug above), `KeyError: torch.int8` (96), `OpConversionFailure` (90), `histogram_cpu` (91).
- **qualcomm** — `Failed to generate Qnn context binary` (542), `pixel_shuffle`/`pixel_unshuffle`
  node errors (131), `min nan should be less than max nan` (52).

## 3. What `--quantize` does at runtime

Executed on the local seams: VGF on Mesa lavapipe + the ML SDK emulation layer, Qualcomm on the
x86 HTP emulator, OpenVINO in-process on the ET host runtime.

**Note on the oracle.** For a quantized job `pregen` replaces the fp32 eager reference with the
**PT2E quantized reference** (the converted graph run on CPU) — the correct oracle, but it means
"MISMATCH disappeared" can mean the reference moved, not that the device improved.

### Headline (all paired graphs)

| backend | n | float OK | int8 OK | float MISM | int8 MISM | float SKIP | int8 SKIP | float CRASH | int8 CRASH |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| vgf | 4,271 | 92.9% | **93.2%** | 108 | 97 | 194 | 193 | 0 | 2 |
| openvino | 3,634 | 79.4% | **77.8%** | 278 | 271 | 461 | 535 | 1 | 0 |
| qualcomm | 5,858 | 86.0% | **80.5%** | 367 | 677 | 408 | 427 | 43 | 40 |

But most of each int8 corpus is **not actually quantized** — `--quantize` is a no-op wherever the
quantizer doesn't annotate. Restricting to graphs whose int8 `.pte` really gained q/dq nodes
(`scripts/ab_quantized_only.py`) is the sharp comparison:

| backend | quantized subset | float OK | int8 OK | delegated float → int8 |
|---|--:|--:|--:|--:|
| vgf | 820 / 4,271 (19.2%) | 93.9% | **95.0%** | 393 → 520 |
| openvino | 279 / 3,634 (7.7%) | 76.7% | **51.3%** | 265 → 279 |
| qualcomm | 1,304 / 5,858 (22.3%) | 83.3% | **68.7%** | **298 → 1,177** |

Three completely different outcomes:

### VGF — net neutral, more coverage
Quantization *helps*: +1.1 pts OK on the quantized subset and 32% more graphs delegated (393→520).
`pixel_shuffle` 55%→79%, `pixel_unshuffle` 82%→96%. Regressions are narrow, 41 graphs total:

- **VGF blob init fails on quantized delegated graphs** — `Failed to process VGF blob` /
  `Init failed for backend VgfBackend: 0x1` at method load. `mean.out` (`w1:46`, fully delegated
  `ops=1` in *both* modes, float OK), `any.dims_out`, `copy`, `sum.IntList_out`, `where.self`.
- **Segfault on quantized blobs** (11): `pixel_shuffle` ×6, `pixel_unshuffle`, `index_select.out`,
  `glu.out`, `ge.Scalar_out`, `le.Tensor_out`.
- **2 new CRASHes on `narrow_copy`** — but `ops=0` in both modes, so this is a **portable** quantized
  kernel abort, not VGF (step-3a ruled out).

### OpenVINO — clear regression, one cause
OK collapses 76.7% → 51.3% on the quantized subset; SKIPs 56 → 126. 74 of the 92 newly-failing
graphs give the **same** error:

```
CALL_DELEGATE execute failed at instruction 0: 0x1
```

The int8 blob compiles AOT and loads, then refuses to execute. All of these are fully delegated
(279/279 of the quantized subset delegate). Worst ops: `round.out` 100%→7.7%, `mul.out` 100%→28.6%,
`bmm.out` 84.6%→53.8%, `pow.Tensor_Scalar_out` 100%→57.1%, `div.out` 100%→68.2%, `mm.out`/`sqrt.out`/
`_native_batch_norm_legit_no_training` 100%→77.8%. Two other classes: `OpenVINO: unsupported` (2) and
a rank-immutability error resizing an out= tensor on `min.dim_min` (2).

### Qualcomm — quantization is what makes QNN delegate, and the delegated kernels are wrong
The mechanism matters more than the number: in float mode QNN's partitioner **rejects almost
everything** on single-op graphs (298 of 1,304 delegated) so the portable CPU kernel computes the
right answer; under `--quantize` the QDQ path makes QNN absorb them (**1,177 of 1,304**). The
regression is not quantization noise — it is the QNN kernel running for the first time.

Of the 480 float-OK → int8-non-OK graphs, step-3a splits them 234 MISMATCH + 67 SKIP + 2 CRASH on
**delegated** graphs vs 123 MISMATCH + 54 SKIP on portable fallback. Confirmed delegated findings:

| op | n | int8 outcome | evidence |
|---|--:|---|---|
| `log10.out` | 33/33 | non-finite mismatch | `log10(0)`: reference `-inf`, device **`-56896.0`** — saturates instead of propagating. Float path was portable (`ops=0`) and correct; int8 delegates (`ops=2`) and is wrong. |
| `log2.out` | 30/30 | non-finite mismatch | same class |
| `acos.out` | 11 | non-finite mismatch | same class |
| `max.dim_max` | 21 | wrong value, median Δ=6 | int64 indices output: reference `4`, device **`-1`** |
| `min.dim_min` | 16 | wrong value, median Δ=7 (+1 crash) | same class |
| `slice_scatter` | 10 | wrong value, median Δ=1.12 | |
| `avg_pool2d.out` | 29/29 | SKIP | delegated blob won't execute |
| `any.dims_out` | 28 | 17 SKIP + mismatch | |
| `scatter.value_out` | 19 | SKIP | |

Ruled out by step 3a (`ops=0`, portable kernel ran): `convolution` (30, median Δ=0.022),
`_native_batch_norm_legit_no_training` (13), `unbind_copy.int`, `where.self`, `split_copy.Tensor` —
all with Δ ≈ 0.005–0.02, i.e. **requantization rounding** between the ET portable quantized kernels
and the torchao reference, not backend bugs. The overall new-mismatch median Δ is 0.02, so the
gross failures in the table above are the signal and the long tail is rounding.

## 4. Confidence and gaps

- **Reproducibility**: the VGF float run was executed twice end-to-end and returned bit-identical
  tallies (3969 / 108 / 194 / 0).
- **Determinism gate**: the QNN `log10.out` headline is **CONFIRMED** — `w0:192` re-fed through 5
  *separate* feed processes (the feeder collapses same-job repeats within one process) returned
  MISMATCH 5/5, on top of 33/33 independently generated `log10` graphs across all 16 workers.
  The other rows have *not* been through a formal N≥5 same-`.pte` gate — treat them as PLAUSIBLE.
- **Corpus sizes** are 4–6k paired graphs per backend, not the 75–100k of the full single-op runs,
  because both fleets of each pair had to run concurrently on a shared box. Per-op sample counts are
  ~10–35, so per-op percentages are directional; the op-level *verdicts* (33/33, 30/30, 29/29) are not.
- **Two openvino float jobs timed out** at the end of the run (10 of 3,634) — inflight at teardown,
  not a backend signal.
- `xnnpack` and `vulkan` were not tested: prior sweeps show they are structurally incapable of
  quantizing a single-operator corpus (XNNPACK annotates fusion *patterns*, Vulkan is weight-only
  and the fuzzer emits no weights).
