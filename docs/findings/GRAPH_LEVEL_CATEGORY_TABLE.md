# Graph-level divergence — multi-operator corpora (post-audit, 2026-08-23)

Companion to `CONSISTENCY_CATEGORY_TABLE.md` (single-operator). Every number here has been re-measured
by a full audit pass; the six audit reports are the authority:

- `vgf/ZEROED_AUDIT.md` · `vgf/VALUE_AUDIT.md` · `vgf/NONFINITE_AUDIT.md` · `vgf/shape_probe/README.md`
- `coreml_mac2/AUDIT.md` · `../findings_v2/qnn_emulator/AUDIT.md`
- `MULTIOP_SMALL_CLAIMS_AUDIT.md` (Vulkan, Ethos-U, XNNPACK, portable)
- `../findings_v2/ethos_u_fvp/REQUANT_SCALE_REVIEW.md`

**Result of the audit: there is exactly ONE defensible graph-level category.** Every other proposed
category either collapsed or turned out to be this one mislabelled. The category is also *larger*
than originally published, while everything else shrank toward zero.

## The one category

### Storage-planning divergence — collapsed outputs / dropped store

**Definition.** A graph returns several tensors; the backend's compiler collapses two of them onto one
piece of storage, writes it once, and the output that received no write is returned in its
zero-initialised state. The operator is correct when it is the graph's only output.

**Why it is one category across backends: the same phenomenon, three source-pinned mechanisms.**

| backend | what collapses | where the write is lost | evidence |
|---|---|---|---|
| **VGF** | duplicate / identity-equivalent producers → one alias-group backing | converter's IO aliasing; the delegate copies back memory the pipeline never wrote | minimal trigger `n0=relu(L0); n1=relu(L0); return (n0,n1)` → out0 all zeros, 3/3, fp32+fp16 |
| **CoreML** | two returned outputs aliasing the same tensor → one buffer | inside the compiled CoreML model | AOT header emits the same `outputName` twice: **49.5% of ZEROED vs 1.4% of device-OK** (35x enrichment) |
| **Vulkan** | two graph-output edges → deduplicated `all_input_nodes` | blob declares **1** output for **2** call-site slots; `output_offset = args.size() - num_outputs` leaves the leading slot unwritten | `prod(acos(zeros)).half() = 2.466796875` == the device-reported `2.4668`; emptying `redundant_ops` removes it |

Chain for Vulkan, fully pinned: `RemoveRedundantOpsTransform` deletes the `clone` →
`process_output_node` iterates the deduplicated node list (`vulkan_graph_builder.py:442`) →
`VulkanBackend.cpp:821` writes the wrong slot. Fires for `clone`, `alias_copy` and shape-preserving
`expand_copy` only, and only when two graph-output edges collapse — which explains the reported
output-set sensitivity.

**ExecuTorch's memory planner is ACQUITTED — measured twice, not argued.**
- VGF: re-running the plan probe **with liveness intervals** gives **0 / 285** overlaps (the published
  "~50% aliased onto a live buffer" reproduces at 31/60 only because the probe has no liveness test).
- Vulkan: forcing the `verify_storage_reuse` gate open (the default `MemoryPlanningAlgorithmSuite`
  instance makes `_callable_name(algo) == "greedy"` false, so it never runs — traced at 0 calls) makes
  the verifier **run and pass**; `allow_overlapping_allocations=False` works.

What remains ExecuTorch's fault is narrow and structural: **nothing checks that a delegate wrote every
output slot it was given.** Inputs and outputs reach `BackendInterface::execute` as one
undifferentiated `Span<EValue*>`, and the runtime's own comment says it "cannot know from the
arguments which are the inputs and which are the outputs". A post-condition is impossible as the
interface is shaped.

### Counts

| backend | defensible | as published | basis |
|---|--:|--:|---|
| VGF | **~308** | 289 | 243 of 289 ZEROED + **65 mislabelled `NONFINITE`** (+ an unmeasured share of 76 blank rows) |
| QNN (HTP) | **~400** | ">=1,644" | 35.0% of 2,936 confirmed sibling-dependent [CI 29.9-40.1], of which 39% all-zero |
| CoreML | **~176** | 204 | 204 - 25 not-delegated - 3 that fail on portable too |
| Vulkan | **~7% of jobs** | "6 A/B" | ~3-4% of delegate output slots, corpus-wide measurement |
| Ethos-U | **1 candidate** | 2 | only `w25:933` survives |
| XNNPACK | **0** | 0 | now *measured*: 325 jobs → 58 OPERATOR / 2 GRAPHOPT, both the generator confound |

VGF's category is **larger** than published, because 65 rows filed as `NONFINITE` are dropped stores
(`mech()` tests NONFINITE *before* ZEROED, so a dropped store whose reference held a nan is
mislabelled), and most sampled blank rows re-run as ZEROED.

**portable was likewise never bisected** — `findings_v1/portable/` has no `graphopt` or `bisect`
artifact either, so it has no measured graph-level row here. `findings/multiop_small_claims_probes/
xnn_graphopt2.py` runs unchanged against `portable` + `corpus_v1/portable` if a number is wanted.

**XNNPACK's zero is now real evidence, and it has a mechanism**: its serializer *rejects*
(`xnnpack_preprocess.py:76`) the exact partition shape that silently drops a store in Vulkan. But note
the "0" was previously never measured — no `bisect_results.tsv`, no `BISECT_BACKEND=xnnpack`, the
bisection tooling postdating the run by 2-5 days. Any earlier text calling it "a result, not a gap"
was unsupported.

## Categories that did NOT survive

| proposed category | verdict | measurement |
|---|---|---|
| **Numeric plan / number format** | **refuted** | VGF: of 21 genuinely-numeric survivors, **0 at fp16 magnitude** (<=1e-3), 0 in 1e-3..1e-2, all 21 above 1e-2 (median ~0.3). Ethos-U requant: 1 of 3 real, and its sign cannot come from a positive scale. CoreML SCALE: **81% vacuous** |
| **Shape / metadata plan** | **ruled out** | 0 of 12. Plan declares the right shape (24/24 lowerings); device emits the right byte count (12/12) |
| **Non-finite** | **not a category** | 34 of 194 survive (17.5%); 65 are ZEROED mislabelled, 46 reproduce on *portable* (often bit-identically), 12 have no delegate, 20 unfalsifiable. Even the 34 are suspect: 26 target an op singular at 0 (`log*` x13, `reciprocal`, `rsqrt`, `fmod`) — whole-output replacement seen through a singular function |
| **Compile/execute failure** | keep, but separately | QNN x86 AOT `prepare` heap corruption is real and well-controlled (`w11:772` aborts in a used process, `READY` 3/3 fresh). Not a divergence — report as toolchain robustness |

The VGF `VALUE` bucket resolves to **~106 of 332** rows that are VGF-specific *and* delegated (not
275), of which ~51 are residual-numeric — and its survivors are not precision at all but wrong
operand / wrong branch under fusion (`atan2` sign-of-pi flip, pi/4 -> pi/2, softmax collapsing to
uniform). Worth one sentence, not a category.

## Ten methodological defects found (the reusable result)

Ranked by how much they changed the numbers. **Every unguarded similarity test was substantially
vacuous; every positive test survived.**

1. **Cross-backend sharing never filtered.** VGF `VALUE`: 47/93 reproduce on portable and/or xnnpack,
   44 bit-identically. VGF `NONFINITE`: 46/194 reproduce on portable.
2. **Portable-fallback (`delegated_ops == 0`) never filtered** on graph-opt populations, though the
   single-op methodology treats it as filter 3a: 23/93 VALUE, 12/194 NONFINITE, 25/204 CoreML ZEROED.
3. **`cone_localizer` bakes ancestors, so it never tests them.** 41 of 46 VGF COMPOSITIONAL ZEROED
   cases have an already-broken ancestor (led by `aten.fill`, which zeroes as a *standalone* one-op
   graph). The SIBLING axis is immune — it does no baking.
4. **Vacuous similarity tests, four independent instances.** SCALE accepts a constant reference or a
   single surviving ratio (Ethos-U 2/3, CoreML 38+4 of 52); ALIAS/CLOBBER accepts `numel<8` or
   multiple matches (VGF 22/31); QNN's "byte-identical" clobber probe uses a **5% relative tolerance**
   with prefix truncation and cannot fail (`w103:514` matched nothing and was still assigned).
5. **Test ordering.** `NONFINITE` is evaluated before `ZEROED`/`ALIAS` → 65 dropped stores mislabelled.
6. **`mech()` tests `isfinite` but never `isnan`** → `nan<->inf` and `+-inf` sign flips fall through to
   `VALUE`.
7. **Harnesses discard the returned job id** (`deep_graphopt.py:57` et al.). With 145/225 timeouts on
   Ethos-U, late results land on the **wrong job** — proved: `w78:202` and `w87:268` share
   byte-identical device values that are `w78:202`'s own dequantized input.
8. **Single-rep A gates.** VGF's 4 `A_ALSO_DIVERGES` exclusions all pass A cleanly at 3 reps.
9. **Stage-conflated counts.** Ethos-U "21 DRIFT + 5 CLOBBER" is the pinpoint stage, not mechanisms (3).
   QNN "`fill.Scalar` 520" is the all-mismatch count (GRAPHOPT ~455). CoreML publishes composition
   statistics computed on a 50-row population under a 204-row headline.
10. **A population was destroyed.** `characterize_graphopt.py` overwrites its TSV in place, so CoreML's
    109-row snapshot — the basis of every §4 statistic — no longer exists.

Residual bookkeeping: VGF's 29 `?` rows are `B_NOREPRO` by construction and 0/12 reproduce → remove
them (984 → 955). The 76 blank rows are the exception path (`ERR`/`DEV:TIMEOUT`), *unmeasured rather
than unreal* — 11/12 sampled reproduce 3/3, mostly as ZEROED.

Clean negatives worth reporting: `compare.py::select`'s silent-fallback hazard does **not** fire
(0 fallbacks in 465 VGF runs; structurally excluded on QNN). Non-finite **leaf injection** is not a
confound in corpus_v4 (`_NONFIN_P = 0.0005`, and 0/194 leaves are non-finite). fp16 is not the QNN
explanation (only 3.8% of confirmed cases within 10x tolerance). The int64 `2^63` sentinel is real but
bounded (QNN 1.0% observed; VGF 19/93 VALUE cases, all with a `bitwise_*_shift` trigger and **19/19
shared**, i.e. not a backend bug).

## Headline for the paper

Multi-operator corpora reveal **one** graph-level defect class: a backend collapses two graph outputs
onto one buffer and leaves the other unwritten. It is confirmed on **four** backends (VGF, CoreML,
Vulkan, QNN) with **three independently source-pinned mechanisms** in three different code layers, it
is silent, and XNNPACK is a genuine measured control that avoids it by rejecting the partition shape.
ExecuTorch's memory planner is exonerated by direct measurement on two backends; what the framework
lacks is any post-condition that a delegate wrote the outputs it was handed.

The second contribution is methodological: of the categories originally proposed, only the one backed
by a *positive* test survived. Every ratio, similarity, or name-heuristic test was substantially
vacuous, and two filters the single-operator methodology applies rigorously (portable fallback,
cross-backend sharing) had never been applied to the graph-level populations at all.
