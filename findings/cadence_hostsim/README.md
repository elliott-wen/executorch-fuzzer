# Cadence (Xtensa HiFi) — single-operator analysis

Backend: **cadence** (ExecuTorch in-tree Cadence backend, int8 `quant=ALWAYS`).
Runtime: **host x86 generic reference kernels** via `cadence_client` — `cadence_runner_io`, the
ExecuTorch runtime linked with `backends/cadence/generic/operators/*.cpp`, so the `.pte`'s
`cadence::` custom ops execute natively on the CPU. No Xtensa toolchain / `xt-run` / DSP needed.
Local broker on `15554/55/56`, 128 host workers.
Corpus: `corpus_v5/cadence`, **84,581** single-operator graphs (`--nodes 1`), 84.5% pregen yield.
Method: [analysis_single.md](../../analysis_single.md), with the **pass-based** filter-3a
adaptation described below. Every verdict runtime-verified.

Run date 2026-09-08. Artifacts in [_work/](_work/).

## Feed tally (full corpus)
| status | count | % |
|---|--:|--:|
| OK | 80,870 | 95.61 |
| MISMATCH | 649 | 0.77 |
| CRASH | 10 | 0.01 |
| SKIP | 3,052 | 3.61 |
| TIMEOUT | 0 | 0 |

84,581 jobs, ~45 s wall at 128 workers (~1,900 jobs/s).

## The decisive filter: `delegated.ops` is meaningless here

Cadence is **pass-based, not a delegate** — `delegated.ops == 0` on all 84,581 jobs, so filter 3a
cannot be applied as written (same situation as `cortex_m`). The honest substitute is: **does the
`.pte` actually call a `cadence::` kernel?** Measured by decoding every `.pte` in the corpus and
grepping its operator table (`_work/scan_shard.py`, results in `_work/pte_cadence_ops.tsv`):

**1,066 of 84,581 graphs (1.26%) reach a `cadence::` compute kernel.** Six kernels are ever
reached, and their outcomes separate perfectly:

| `cadence::` kernel | graphs | OK | MISMATCH | SKIP |
|---|--:|--:|--:|--:|
| `transposed_convolution` | 266 | 0 | 0 | **266** |
| `quantized_matmul` | 249 | **249** | 0 | 0 |
| `quantized_relu` | 233 | **233** | 0 | 0 |
| `quantized_linear` | 228 | 1 | **227** | 0 |
| `avg_pool2d` | 62 | 0 | 0 | **62** |
| `conv1d` | 28 | 0 | 0 | **28** |

Why coverage is this thin: the Cadence quantizer's fusion patterns (linear+bias, matmul, conv+relu)
mostly **cannot form in a `--nodes 1` corpus**, so 98.7% of graphs lower to pure portable ATen. The
1.26% here is not a Cadence op-support limit — it is a corpus-shape limit, and it means conv,
pooling, softmax, layer/group-norm and the elementwise family are **untested on this backend**.

**3a-equivalent split** — of the 649 MISMATCHes, **227 ran a `cadence::` kernel** and **422 are
portable-fallback** (`remainder` 210, `tril` 74, `_native_batch_norm_legit` 51, `div.Scalar` 38,
`native_group_norm` 20, `fmod` 18, `pow` 7, others 4). Those 422 are portable-runtime bugs
reachable from every backend, not Cadence bugs, and are excluded. **All 10 CRASHes are likewise
portable** (`narrow_copy` ×8, `unfold_copy`, `max_pool2d_with_indices_backward`; each reproduces at
`--window 1`, all `tensor_impl.h:136 assert failed (dim < dim_ && dim >= 0)`) — zero CRASHes touch a
`cadence::` kernel.

## The one confirmed Cadence bug

### `addmm`'s `beta` and `alpha` are silently dropped by the quantized-linear fusion
- **227 / 228** graphs that reach `cadence::quantized_linear` MISMATCH. All **227 CONFIRMED
  deterministic** (5/5 independent re-runs; `_work/determinism_r1.jsonl` + 4 further rounds).
- The separation is exact: the **single OK graph is the only one in the corpus with
  `beta=1, alpha=1`**. Every one of the 227 failures has `(beta, alpha) != (1, 1)`; not one
  `(1,1)` graph fails.
- `max|delta|` is O(1)–O(30) against an int8 reference whose own error is ~0.05 — three orders of
  magnitude past quantization noise, and past the `rtol=0.01/atol=0.001` gate on all 227.
- **Proof of mechanism.** Seven distinct jobs share byte-identical leaf tensors and differ *only*
  in `(beta, alpha)`. The runtime returns the **same output tensor for all seven**, including for
  pairs whose correct answers are exact negations of each other:

  | job | beta | alpha | \|et − (bias + mat1@mat2)\| | \|et − (β·bias + α·mat1@mat2)\| | \|eager − correct\| |
  |---|--:|--:|--:|--:|--:|
  | w102:645 | −5 | 4 | **0.0245** | 7.2132 | 0.0588 |
  | w109:62 | 5 | −4 | **0.0245** | 9.0402 | 0.0588 |
  | w14:160 | −4 | 4 | **0.0245** | 6.1028 | 0.0408 |
  | w19:684 | 4 | −4 | **0.0245** | 10.1505 | 0.0408 |

  The runtime computes `bias + mat1@mat2` to within int8 error while the correct
  `beta*bias + alpha*(mat1@mat2)` is off by 6–10. Both scalars are discarded.
- **Root cause** — `backends/cadence/aot/quantizer/patterns.py`, `AddmmPattern`:
  `partition_types()` claims `aten.addmm.default` unconditionally; `get_anchors()` maps
  `args[1]→input`, `args[2]→weight`, `args[0]→bias`; `fuse()` reads only those three and emits
  `cadence::quantized_linear.per_tensor`, which computes `input @ weight.T + bias` and has no
  scalar multipliers. **Neither method ever reads `beta` or `alpha`.**
- **The guard exists on the parallel path.** The *unquantized* rewrite
  `ReplaceAddMMWithLinearPass` (`aot/replace_ops.py:343`) does check —
  `fit_bias = beta == 1.0`, `fit_mat2 = alpha == 1.0`, `if not (fit_bias and fit_mat2): return False`
  — and declines to rewrite. (Its own `beta != 1.0` / `alpha != 1.0` rescaling blocks at
  `replace_ops.py:389-421` are therefore unreachable dead code.) Cadence runs `quant=ALWAYS`, so
  production traffic takes the **unguarded** quantizer path.
- **Category: Comp.** The output is structurally valid (right shape, right dtype) and non-zero, but
  invariant to the `beta`/`alpha` operands — the table's tie-break for
  "non-zero but invariant to the operand".

## Second defect (not a consistency row): three `cadence::` kernels can never load

**356 of the 1,066 cadence-kernel graphs (33.4%) are undeployable** — the AOT passes emit an op that
no runtime, host or `xt-run`, can register:

```
kernel 'cadence::transposed_convolution.out' not found.   266 graphs
kernel 'cadence::avg_pool2d.out' not found.                62 graphs
kernel 'cadence::conv1d.out' not found.                    28 graphs
```

These ops have generic C++ implementations **and** are emitted by `aot/replace_ops.py`, but are
absent from `aot/functions.yaml`, so codegen never registers them. This is a self-inconsistency in
the backend: it lowers to operators its own runtime does not provide. It surfaces as SKIP, so the
consistency table excludes it, but it is the largest single loss of deployable Cadence coverage in
the corpus — bigger than the numeric bug.

## SKIP taxonomy (3,052) — mapped onto [RUNTIME_SKIP_TAXONOMY.md](../RUNTIME_SKIP_TAXONOMY.md)

| id | n | what |
|---|--:|---|
| **RG** | 1,705 | portable kernel validation guards, cause named: `normalization_ops_util.cpp:160 (in.size(0) == N)` 658 · `index_util.cpp:24 (index.scalar_type() == Long)` 624 · `op_copy.cpp:33 (non_blocking == false)` 227 · `op_arange.cpp:55` 63 · `copy_ops_util.cpp:165 (implicit == false)` 58 · `tensor_util.h:411` 47 · `kernel_ops_util.cpp:389` 21 · 7 others |
| **RH** | 991 | client output-decode refusal: the lowered program's output dtype differs from the eager reference (`masked_fill.Scalar` 432, `fill.Scalar` 382, `tril.out` 130, `mul.Scalar` 45, `add.Scalar` 2). See below |
| **RM** | 356 | the three unregistered `cadence::` kernels above |

Full per-job reasons in `_work/rc2_reasons.tsv` (every rc=2 job re-run to recover the runtime's own
message, `_work/why_skip.py`).

**The 991 RH rows are a real dtype divergence, currently unclassifiable.** The Cadence quant passes
promote the output dtype of `masked_fill`/`fill`/`tril`/`mul.Scalar`, so the `.pte` writes 2× or 4×
the bytes the eager meta expects and the client refuses to reinterpret them (correctly — an
unchecked reinterpret would fabricate a tensor to diff). That is a **Meta**-class defect by
definition, but it lands as SKIP rather than MISMATCH, so no operator can be counted. All 991 are
portable-fallback graphs, so it is a whole-graph pass artifact, not a kernel bug. Classifying it
needs a decode path that reads dtypes off the `.pte` — the highest-value harness follow-up, and
plausibly worth more Meta operators than any other backend row.

## Consistency-category row

| backend | device reported | NumSem | Comp | Map | Meta | total ops | root causes |
|---|---|--:|--:|--:|--:|--:|--:|
| Cadence (Xtensa HiFi) | host generic reference kernels | 0 | **1** | 0 | 0 | **1** | 1 |

One operator overload: `addmm.out` → `cadence::quantized_linear`. One mechanism.

## Scope: this is the AOT path, not the DSP

The generic reference kernels validate **AOT lowering** (quantize → fuse → `cadence::` rewrite →
execute); they are not the hand-vectorised HiFi4 microkernels that ship on the DSP. The kernel-level
oracle is the separate generic-vs-HiFi4 differential under `xt-run` (21,365 jobs, 527 actionable,
`tmp/cadence_diff_full.log`) — a **different** oracle answering a different question, and its
findings (`fmod`, `mean`/`_softmax` crashes, `rsub`, `div`) do not appear here because the two
targets share this same reference implementation on the generic side.
