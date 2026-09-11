# Audit — the four smaller multi-op graph-optimization claims (A–D)

Scope: `findings_v1/vulkan_galaxy_26` + `findings_v2/vulkan_*` (Claim A), `findings_v2/ethos_u_fvp`
(Claim B), `findings_v1/xnnpack_x64` + `findings_v1/xnnpack-arm` (Claim C),
`findings_v1/portable` (Claim D). VGF / CoreML / QNN were audited separately and are untouched here.

Everything below was run on this Linux host with
`PYTHONPATH=/data/jwen929 CUDA_VISIBLE_DEVICES= MOBILE_BACKENDS=<one backend>`, python
`/data/jwen929/mobile/.venv/bin/python`. Every probe script and its raw output are checked in under
**`findings/multiop_small_claims_probes/`**: `vk_plan_probe.py`, `vk_gate.py`, `vk_alias.py`,
`vk_cause.py`, `vk_ops{,2,3}.py`, `vk_scan.py`, `vk_scan_sum.py`, `eu_probe.py`, `eu_scale.py`,
`alias_probe.py`, `xnn_graphopt2.py`, `xnn_diag.py`, `portable_shape2.py`, `portable_shape3.py`,
plus `xnn_graphopt.tsv`, `vk_scan_sum.log`, `vk_scan_progress.log`. The three corpus scans were
stopped at the sample sizes reported below (they take ~5 s/job of Vulkan lowering); re-running a
probe with a larger `N` extends the sample without changing the setup.

**No Vulkan phone and no Corstone-300 FVP is available** (`FVP_Corstone_SSE-300_*` and
`arm-none-eabi-gcc` are not installed; `pytorch_ref/executorch/examples/arm/ethos-u-scratch/`
is empty). All Vulkan and Ethos-U claims below are therefore audited **AOT / host-side only**;
no device number was re-measured. Where that matters it is called out.

The installed ExecuTorch (`.venv/lib64/python3.12/site-packages/executorch`) is byte-identical
to `pytorch_ref/executorch` for every file cited (`remove_redundant_ops.py`,
`vulkan_preprocess.py`, `exir/memory_planning.py`, `exir/passes/memory_planning_pass.py`),
so source claims apply to the tree that produced the corpora.

**Provenance note.** Section C.1 (the "was XNNPACK ever bisected?" evidence) was originally
drafted from a delegated repo search whose result never came back. Every claim in C.1 has since been
**re-verified directly** by the commands shown; three details were wrong in the first draft and are
corrected here — the `localized.jsonl` field list (it also carries `node`/`detail` on 9,731 of
10,988 records, and `crashes.jsonl` has an unrelated crash-signal `verdict` key), and the
cross-backend "control" file (`findings/vgf/xbackend.tsv` is 42 rows = 7 jobs x 6 backends, not
7 rows). Nothing in the C.1 conclusion changed.

---

## HEADLINE 1 — Claim A: the mechanism in the write-up is WRONG; the real root cause is now
## fully source-pinned and predicts the device number exactly

### A.1 `verify_storage_reuse` gate — CONFIRMED closed, and forcing it open ACQUITS the planner

`exir/passes/memory_planning_pass.py:293-300`:

```python
verifier.verify_graph_input_output()
if (callable(self.memory_planning_algo)
        and _callable_name(self.memory_planning_algo) == "greedy"):
    verifier.verify_storage_reuse()
```

Measured (`vk_gate.py`):

| expression | value |
|---|---|
| `_callable_name(MemoryPlanningAlgorithmSuite())` | `'<executorch.exir.memory_planning.MemoryPlanningAlgorithmSuite object at 0x…>'` |
| `_callable_name(partial(greedy, allow_overlapping_allocations=False))` | `'greedy'` |
| gate opens for the suite instance? | **False** |

`backends/vulkan/vulkan_preprocess.py:221-233` (confirmed verbatim) wraps the partial in a
`MemoryPlanningAlgorithmSuite` **instance**, so `_callable_name` falls through to `str(obj)` and
the gate never opens. A traced `Verifier.verify_storage_reuse` confirms it: **as shipped it is
never called** for any Vulkan lowering (0 calls across 4 graphs). Note `partial` *is* handled by
`_callable_name`, so the same code passing the bare partial *would* open the gate — the suite
wrapper is what silences the check.

**The decisive experiment.** I replaced `MemoryPlanningAlgorithmSuite` in `vulkan_preprocess` with a
factory returning a plain function literally named `greedy` that delegates to the real suite
(so behaviour is unchanged but `_callable_name(...) == "greedy"`). Result:

```
>>> GATE FORCED OPEN
=== Bug (lift_fresh_copy)   lowering OK   verify_storage_reuse(allow=False) -> 0 pairs / 1 pair
=== BugClone (clone)        lowering OK   verify_storage_reuse(allow=False) -> 0 pairs / 1 pair
=== Control (add 0.0)       lowering OK   verify_storage_reuse(allow=False) -> 1 pair  / 1 pair
=== Big (clone + expand_copy + 4 live siblings)  lowering OK  -> 0 pairs / 2 pairs
```

**`verify_storage_reuse` does NOT raise** on any of the repro graphs, including the `clone` /
`lift_fresh_copy` ones. Per the task's own decision rule: *the plan is a legal reuse.* The
"planner models the source as dead after the copy and reuses its live storage" story is
**not supported**.

### A.2 The plan does not overlap the outputs either (negative result)

`vk_plan_probe.py` dumps every `TensorSpec`'s `(mem_id, mem_offset, allocated_memory, lifetime)`
from the delegate-internal program right after the Vulkan `MemoryPlanningPass`. For `Bug`/`BugClone`
the delegated `sigmoid` subgraph is:

```
x                     mem1 @16  16B  lt=(0,1)
aten_sigmoid_default  mem1 @0   16B  lt=(1,2)
```

Two tensors, disjoint offsets, no overlap at all. Every "overlap" my first pass printed was the
same `TensorSpec` object reached twice (via the node and via the `output` node) — `dedup=True` in
`collect_specs_from_nodes` removes those, which is why `verify_storage_reuse` reports 0 pairs.
**There is no clobbering overlap in the AOT plan.** `allow_overlapping_allocations=False` is
confirmed present, and it is doing its job.

### A.3 What actually happens: the delegate declares FEWER outputs than the call site passes

`vk_alias.py` dumps the serialized `VkGraph` alongside the outer `.pte`:

| graph | `VkGraph#0` inputs / outputs | outer `DelegateCall` args | output slots | verdict |
|---|---|---|---|---|
| `Bug`/`BugClone` (`clone`) | `[0]` / **`[1]`** (1 output) | `[0, 1, 2]` | **2** | **1 slot never written** |
| `Control` (`add 0.0`) | `[2]` / `[3, 5]` (2 outputs) | `[0, 1, 2]` | 2 | OK |
| `DirectAlias` (`clone`, 3 outputs) | `[2]` / `[3, 4]` (2 outputs) | `[0,1,2,3]` | 3 | **1 slot never written** |

The delegate-internal graph's output list for `Bug` is literally
`['aten_sigmoid_default', 'aten_sigmoid_default']` — the same node twice, because
`RemoveRedundantOpsTransform` deleted the `clone` and `replace_all_uses_with(src)` collapsed both
graph outputs onto `n0`. Then
`backends/vulkan/serialization/vulkan_graph_builder.py:441-454`:

```python
def process_output_node(self, node: Node) -> None:
    for out_node in node.all_input_nodes:          # <-- DEDUPLICATES
        ...
        self.output_ids.append(self.node_to_value_ids[out_node])
```

`fx.Node.all_input_nodes` is de-duplicated, so `output_ids` gets **one** entry for **two**
positional graph outputs. At runtime, `backends/vulkan/runtime/VulkanBackend.cpp:821`:

```cpp
const size_t output_offset = args.size() - num_outputs;   // 3 - 1 = 2
for (size_t i = 0; i < num_outputs; i++) { ... args[output_offset + i] ... }
```

so only `args[2]` is written. `args[1]` — the buffer the outer program then feeds to
`aten::acos.out` (outer `KernelCall aten::acos.out args=[1,3,3]`) — is **never written** and is
read as whatever the arena holds.

### A.4 The mechanism predicts the reported device value exactly

```
prod(acos(zeros(2)).half())        = 2.466796875      <-- device-reported out[0] = 2.4668
prod(acos(sigmoid(x)).half())      = 0.8486328125     <-- eager / correct out[0] = 0.8486
```

A zero-filled unwritten buffer reproduces `2.4668` to every printed digit. It also explains why
`out[1]` (`acosh`, fed by `args[2]`, which *is* written) was not the reported failure, and why
`Control` is clean — `add(n0, 0.0)` is not in `redundant_ops`, so nothing collapses and the blob
declares 2 outputs. **`add` is confirmed absent from the set.**

`redundant_ops` (`backends/vulkan/_passes/remove_redundant_ops.py:25-39`) is confirmed **exactly** as
claimed: `torch.clone`, `aten.clone.default` (aten + edge), `aten.alias.default` (aten + edge),
`aten.alias_copy.default` (aten + edge), `edge.aten.lift_fresh_copy.default`,
`edge.dim_order_ops._to_dim_order_copy.default`, `edge.dim_order_ops._clone_dim_order.default`,
`edge.aten.expand_copy.default`, `edge.aten.copy.default`. Rewiring semantics confirmed:
`node.replace_all_uses_with(node.args[src_idx])` + `eliminate_dead_code()`, guarded by
`_should_remove` requiring identical dtype **and identical shape** — so a genuinely *expanding*
`expand_copy` is **not** removed (relevant to A.6).

### A.5 Causal confirmation

`vk_cause.py` — same graph, only `RemoveRedundantOpsTransform.redundant_ops` emptied:

```
as shipped : declared_inputs=1 declared_outputs=1 callsite_args=3 -> 2 slots  *** SHORTFALL 1 ***
neutralised: declared_inputs=1 declared_outputs=2 callsite_args=3 -> 2 slots  OK
```

The shortfall is caused by the copy-elision pass, and by nothing else.

### A.5b Which ops, and which topologies, actually trigger it (`vk_ops.py`, `vk_ops2/3.py`)

Same probe, sweeping the copy-ish ops across three topologies. "UNWRITTEN" = the blob declares
fewer outputs than the `DelegateCall` passes slots for.

| op (co-returned with the source) | `f(n0), mul(n0,2)` | `f(n0), n0, mul(n0,2)` | `prod(acos(n0)), acosh(f(n0))` (partition boundary) |
|---|---|---|---|
| `clone` | ok | **1 UNWRITTEN** | **1 UNWRITTEN** |
| `alias_copy` | ok | **1 UNWRITTEN** | **1 UNWRITTEN** |
| `expand_copy` **same shape** | ok | **1 UNWRITTEN** | **1 UNWRITTEN** |
| `expand_copy` **growing** | ok | ok | ok |
| `view_copy` / `transpose_copy` / `permute_copy` / `squeeze_copy` | ok | ok | ok |
| `slice_copy` (full) / `.to(same dtype)` | ok | ok | ok — *call-site* arity drops too |
| `add(n0,0.0)` / `mul(n0,1.0)` (controls) | ok | ok | ok |

Three things follow:

1. **Only the ops that `RemoveRedundantOpsTransform` actually elides trigger it** — `clone`,
   `alias_copy`, and a **shape-preserving** `expand_copy`. A *growing* `expand_copy` is not in
   scope of `_should_remove` (shape guard) and is clean. So the artifact's `expand_copy` 4/4
   instances **are** covered by this mechanism **iff** those `expand_copy` calls were shape-
   preserving (the generator emits many such no-op expands); a growing one would need another story.
   `transpose_copy`/`view_copy`/`permute_copy` are **not** covered — consistent with the artifact
   already calling `transpose_copy` flaky.
2. **Topology matters**: the shortfall needs two distinct graph-output edges to collapse onto one
   internal value. Co-returning the source itself, or having two *separate* consumers on the far
   side of the partition boundary (the `Bug` shape), does it; one output plus an unrelated sibling
   does not. This is exactly why the artifact found the corruption "sensitive to the output set".
3. **Elision before partitioning is safe; elision inside `preprocess()` is not.** `slice_copy`(full)
   and a no-op `.to()` are removed at the edge level *before* the partition is drawn, so the
   `DelegateCall` arity drops in step and nothing is left unwritten. `RemoveRedundantOpsTransform`
   runs at `vulkan_preprocess.py:172`, i.e. *after* the partitioner has already fixed the call-site
   arity — that ordering is the bug.

### A.6 Corpus-wide rate (host-only, new measurement)

Two independent host-only samples re-lower random `corpus_v2/vulkan` jobs (`quantize=False`, all
lowered READY) and compare `len(VkGraph.output_ids)` against
`len(DelegateCall.args) - len(input_ids)`:

**(a) per-blob** (`vk_scan.py`, seed 1234) — pairs each serialized blob with its `DelegateCall`:

```
jobs 300 (all READY)   delegate blobs 620
blobs with declared_outputs < call-site output slots : 21  (3.4% of blobs)
jobs with at least one such blob                     : 21  (7.0% of jobs)
blobs whose output_ids contain a literal duplicate   :  0   <-- the dedup is why
```

**(b) order-independent** (`vk_scan_sum.py`, seed 4242) — compares, per job,
`sum(declared outputs)` vs `sum(call-site output slots)`, so blob-to-call pairing cannot bias it:

```
jobs 25 (all READY, all with a delegate)
jobs with an output-slot deficit : 4  (16%)
deficit output slots             : 5 / 117 total slots  (4.3% of slots)
```

The two agree to within sampling noise on the *slot* rate (3.4% vs 4.3%); the job rate differs
(7% vs 16%) only because sample (b) is 25 jobs. Report the **slot** rate, and report it as
"~3–4% of Vulkan delegate output slots in a sample of `corpus_v2/vulkan` are never written",
with ~7% of jobs affected, i.e. on the order of 5,000 of `corpus_v2/vulkan`'s 77,069 jobs — a sound
AOT upper bound on how often the mechanism can fire, measurable with no GPU. `dup_outid_blobs = 0` in every sample is itself the
signature: `output_ids` never *contains* a duplicate, because `all_input_nodes` removed it.

This is a *sufficient* AOT condition for an unwritten output buffer and needs no GPU. It is a much
better denominator than the raw 903 GRAPHOPT count, and it is directly falsifiable.

### A.7 Verdict on Claim A

| sub-claim | status |
|---|---|
| (1) observation: a sibling output that never reads the copy is wrong; swapping the copy's producer to `add(n0,0.0)` fixes it | **VERIFIED (as an observation)** — and now *explained*, with an exact numeric match |
| (1) stated mechanism: "copy elided to a buffer alias; planner treats the source as dead and reuses its storage while live" | **REFUTED.** No storage overlap in the plan; `verify_storage_reuse` passes when forced to run; overlapping allocations are disabled and stay disabled. The real fault is an **output-arity shortfall** (`process_output_node` dedup) turning into an **unwritten delegate output slot** in `VulkanBackend.cpp` — a *dropped store*, not a reuse-of-live-storage. |
| (1) "the exact pass/line is not yet pinned" | **superseded** — now pinned to `vulkan_graph_builder.py:442` (`all_input_nodes`) × `RemoveRedundantOpsTransform` × `VulkanBackend.cpp:821` |
| (2) A/B counts `expand_copy` 4/4, `clone` 2/3 | **UNVERIFIABLE NOW** (no device). Additional concern: `tmp/confirm_graphopt.py:51` discards the returned job id (see B.1), so those A/B labels inherit a result-misattribution risk. Refinement from A.5b: a **shape-preserving** `expand_copy` *is* elided and *does* produce the shortfall, so the 4/4 `expand_copy` instances are covered **iff** those expands were shape-preserving; a growing `expand_copy` is clean and would need a separate story. |
| (3) 903 overstates the bug; `index_put` an artifact; `transpose_copy`/`alias_copy`/some `clone` flaky | **plausible and now better supported** — but the *replacement* number should be the AOT arity-shortfall rate (A.6), not a device A/B count. |
| `verify_storage_reuse` gate never runs | **VERIFIED** — a genuine ExecuTorch hygiene defect worth reporting in its own right, but it is *not* the cause of this bug (the plan is legal). |

**Verdict: NEEDS RE-WORDING (mechanism), and the re-wording is a strict upgrade.** The bug is
real, silent, and now has a three-line source chain, a causal AOT experiment, an exact numeric
prediction of the observed device value, and a corpus-wide AOT rate. Every sentence attributing it
to memory planning / lifetime / buffer reuse / the shared ExecuTorch planner must be replaced.
This also **moves Vulkan into the same class as the VGF and CoreML "ZEROED dropped-store"
findings** (unwritten output buffer), which is a cross-backend claim the artifact currently misses,
while *removing* it from the "ExecuTorch memory planner" class.

---

## HEADLINE 2 — Claim C: the XNNPACK zero was never measured. I measured it.

### C.1 Was it ever looked for? NO — verified

A thorough search of `findings_v1/xnnpack_x64/`, `findings_v1/xnnpack-arm/`, `findings/xnnpack_x64/`,
`findings/xnnpack_arm64/`, `tmp/`, and the GRAPHOPT toolchain establishes:

* Neither XNNPACK findings dir contains `bisect_results.tsv`, `operator_isolation.tsv`,
  `skip_log.tsv`, a `WORKFLOW_RESULTS.md`, a `bugs/graphopt-*.md`, or any GRAPHOPT token.
  Compare `findings_v2/vulkan_asus_a12201/bisect_results.tsv` (8,531 rows, `7025 OPERATOR /
  903 GRAPHOPT / 598 NOREPRO / 4 BISECT_CRASH`).
* The localizer that *was* run (`tmp/run_xnnpack/localized.jsonl`, 10,988 records) emits
  `{job, flagged_op, root_op, kind, status, inherited}` plus `{node, detail}` on 9,731 of them —
  a *first-divergence node* attribution with **no output-set verdict field**, structurally incapable
  of expressing OPERATOR vs GRAPHOPT. (`tmp/run_xnnpack/crashes.jsonl` does carry a `verdict` key,
  but its values are `SIGABRT`/`SIGSEGV`/`SIGFPE` — a crash-signal label from the *crash* localizer,
  unrelated to graph-opt.) The strings `OPERATOR` and `GRAPHOPT` appear **nowhere** under
  `tmp/run_xnnpack/`.
* `grep -ic xnnpack` over `tmp/bisect_all.py`, `tmp/bisect_graph.py`, `tmp/confirm_graphopt.py`,
  `tmp/deep_graphopt.py`, `tmp/pinpoint_graphopt.py`, `tmp/pinpoint_clobber.py`,
  `gen/diff/bisect.py`, `gen/diff/cone_predicate.py`, `gen/diff/ddmin.py`: **0 hits each**. The
  shard dirs that exist are `bisect_shards{,_asus,_ethosu,_qnn,_v1_archive}` — none for xnnpack.
  `BISECT_BACKEND=xnnpack` occurs nowhere in the repo except in audit prose.
* Timeline (mtimes, verified): `findings_v1/xnnpack_x64/README.md` and
  `findings_v1/xnnpack-arm/README.md` **2026-06-27**; `tmp/bisect_all.py` **2026-06-29**;
  `gen/diff/{ddmin,bisect,cone_predicate}.py` and `analysis.md` **2026-07-02**. The XNNPACK run
  **predates the two-axis workflow by 2–5 days.**
* The input existed and was never used: `tmp/run_xnnpack/skip_reasons_xnnpack.tsv` has **11,024
  MISMATCH rows** in exactly the format `tmp/bisect_all.py:45-56` consumes. `corpus_v2/xnnpack` was
  generated — 74,601 job sources on disk — and (inferred from the total absence of any bisect
  artifact) never fed through the graph-opt stage.
* **Neither XNNPACK findings dir ever states a zero.** The zero was stated only in
  `findings/GRAPH_LEVEL_CATEGORY_TABLE.md` — in the version current when this audit ran (mtime
  10:24), the row at `:233` plus the narration at `:250-252`: *"XNNPACK's zero is a result, not a
  gap … it was the cross-backend control that proved both the CoreML and VGF dropped stores are
  backend-specific"* — with **no XNNPACK artifact cited**. (That file was rewritten at 10:39, after
  this audit; see the status note below. The line numbers above refer to the pre-rewrite version.) The "control" actually used is
  `findings/vgf/xbackend.tsv` — 42 data rows = **7 `corpus_v4/vgf` jobs × 6 backends**, not
  `corpus_v1/xnnpack` — and `findings/vgf/GRAPHOPT_REPORT.md` itself records that on the non-finite
  case `w102:563` xnnpack *and* openvino also diverge (both saturate `nan → −1.0`). So the control
  is 7 VGF jobs, and xnnpack is not clean on all of them.
* `analysis.md`'s **"No borrowed verdicts"** rule ("A finding for *this* device must come from
  bisecting *this* device's own failing graphs") and its **"Coverage gate"** ("Step 2 is complete
  only when every MISMATCH `job_id` in the skip-log has its own verdict row in a
  `bisect_results.tsv` … assert it") are both violated: XNNPACK has 11,024 MISMATCH rows and
  **0** verdict rows.
* `findings_v1/portable/` has the same gap: the strings `graphopt` and `bisect` appear in
  **none** of its files, yet the category table asserts a `0` for portable on the same non-evidence.

**So the "0" as published is evidence of absence of looking, not absence of bugs.**

### C.2 So I ran the missing experiment

Two independent host tests, both in-process (deterministic, no broker, no job-id ambiguity):

**(a) Targeted aliasing probe** (`alias_probe.py`) — 10 hand-built multi-output graphs of exactly
the aliasing shape (a computed tensor co-returned with `clone`/`alias_copy`/`expand_copy`/
`view_copy` of a shared value, the same node returned twice, an input co-returned, a 4-output
version, and the Vulkan `Bug` shape), each run through `mobile.executor.et_runner.run_pte`:

```
xnnpack : 8 buildable cases, 8/8 ALL OUTPUTS OK.  2 cases refuse to lower (see below).
portable: 10/10 ALL OUTPUTS OK.
```

**No dropped store, no clobbered sibling, no zeroed output — on either backend.**

Notably, XNNPACK **fails safe** on exactly the structure that silently breaks Vulkan.
`backends/xnnpack/xnnpack_preprocess.py:76` raises

> `Output node 'aten_sigmoid_default' is already in the inputs. This is likely due to pass through arguments, which are not supported in XNNPACK Delegate.`

i.e. XNNPACK refuses to serialize a partition whose output collapses onto an existing value,
where Vulkan serializes a short `output_ids` list and drops the store. That contrast is a real
finding and strengthens the "backend-specific" argument far better than an unmeasured zero does.

**(b) The actual output-set bisection on `corpus_v1/xnnpack`** (`xnn_graphopt2.py`, fork-isolated
per job because the corpus is full of native aborts). For every mismatching output: run the target
alone (A) vs the full return set (B), 3 reps each, via `output_set_localizer`.

Sampled `corpus_v1/xnnpack` (fork-isolated, 3 reps per side):

```
jobs attempted                          : 325
  FULL_OK (no mismatch)                 : 159
  execute() RuntimeError (kernel refusal): 108
  native abort (SIGABRT)                :  27
  >= 1 mismatching output               :  31
mismatching outputs bisected            :  60
  OPERATOR (wrong alone too)            :  58
  GRAPHOPT (correct alone, wrong with siblings) : 2
  NOREPRO                               :   0
target-alone side failed to build       :   0
```

Two things to note against Ethos-U: **every** A-side lowered (0 INCONCL, vs Ethos-U's 163/225),
and every verdict was stable over 3 reps (0 NOREPRO, vs 9/19 NO_BREAK) — because this runs
in-process with no broker and no job-id ambiguity.

Both GRAPHOPT candidates are deterministic (5/5 reps in a dedicated re-run, `xnn_diag.py`) and
both are **generator aliased-`out=` confounds**, not memory planning:

* `w31:508` out7 (`n19 = sum.IntList_out(n9, out=empty((), int64))`): the graph contains
  `n1 = max.dim_max(L1, -2, False, max=n0, max_values=n0)` — eager **writes into `n0`**, and `n9 =
  acosh.out(n0)`. Returning the target alone prunes `n1` from the cone, so eager sees an unmutated
  `n0`. Device delta is `9.223e18 = 2^63`, the documented int64-sentinel confound.
* `w9:638` out5 (`n15 = sign.out(n1)`): the graph contains `n12 = log1p.out(n11, out=n1)` — eager
  **writes into `n1`**. Same story; also intersects the documented portable `sign(NaN)` bug.

So: XNNPACK's graph-opt bucket is **not** empty, but every member is the Claim-D generator
aliasing artifact. The substantive conclusion survives; the evidentiary basis has to change.

### C.3 Verdict on Claim C

**UNSUPPORTED AS WRITTEN — must be re-worded.** "XNNPACK's zero is a result, not a gap" is false
as published: the category was never measured on the XNNPACK corpus, no artifact exists, and the
tooling that produces the verdict postdates the run. The defensible replacement, backed by this
audit:

> On a 325-job random sample of `corpus_v1/xnnpack`, an in-process output-set bisection classified
> 60 mismatching outputs as 58 OPERATOR / **2 GRAPHOPT** / 0 NOREPRO, and both GRAPHOPT cases are
> the differential generator's aliased-`out=` in-place-mutation artifact (the same confound as
> Claim D), not a storage-planning bug. A targeted probe of 8 aliasing-shaped multi-output graphs
> returned correct values on every output. Independently, XNNPACK's serializer *rejects* the
> partition shape that silently drops a store in Vulkan (`xnnpack_preprocess.py:76`), so its
> cleanliness here is partly a fail-safe, not just luck.

Two further wording constraints: (i) the sample is 325 of 100,649 jobs — say "sample", not "the
corpus"; (ii) the same re-wording applies verbatim to the **portable** `0`, which rests on the
identical non-evidence (`findings_v1/portable/` contains no `graphopt` or `bisect` artifact).

---

## Claim B — Ethos-U graph-opt, the 7 remaining classified rows

`REQUANT_SCALE_REVIEW.md` (the 3 SCALE rows) was read and **not** redone; its conclusions are
consistent with everything below and I recommend keeping it.

### B.1 A pipeline-wide harness defect contaminates the whole dataset (VERIFIED)

`tmp/deep_graphopt.py:57` (and the same line in every sibling script):

```python
_, st, det, raw = P.decode_result(D.recv_multipart())
```

**The returned job id is discarded.** One DEALER socket is reused for every job, and on timeout
the code `return`s without draining. So a late result for job *N* is accepted as the answer to
job *N+1*. Confirmed present in: `tmp/deep_graphopt.py:57`, `tmp/pinpoint_graphopt.py:73`,
`tmp/pinpoint_clobber.py:47`, `tmp/bisect_all.py:146`, `tmp/bisect_graph.py:84`,
`tmp/confirm_graphopt.py:51`. Contrast
`findings_v1/vulkan_galaxy_26/bugs/repro_vulkan-copy-elision-aliasing.py:65`, which *does*
check (`if rjid != jid: continue`). With **145 of 226** pinpoint rows being TIMEOUT, late frames
are not hypothetical.

This is enough on its own to explain the "flakiness" the artifact itself reports (9/19 NO_BREAK,
and pinpoint-vs-mechanism device values disagreeing for the *same* `(job,out)` — e.g. `w78:202`
out4 pinpoint `[nan, nan, 0.731, 0.731]` vs mechanism `[-1.1619…, -0.1609…, 0.0]`).

### B.2 (i) The identical device values — VERIFIED identical, and they are a harness artifact

Byte-identical across two different jobs and two different target ops (verified by direct
comparison of `graphopt_mechanism.tsv`; no other dev string in the file is duplicated):

```
w78:202  out4  n6  UNSTRUCTURED  dev[:3]=[-1.1619468927383423, -0.16088494658470154, 0.0]  sigmoid.out
w87:268  out2  n7  UNSTRUCTURED  dev[:3]=[-1.1619468927383423, -0.16088494658470154, 0.0]  logit.default
```

What those numbers are: both are exact multiples of one scale,
`s = 0.01787610604212834`, with codes `q = [-65, -9, 0]` — i.e. a **dequantized int8 tensor**.
`-65·s = -1.1619468927383423` and `-9·s = -0.16088494658470154` are the quantize→dequantize of
`-1.158203125` and `-0.166015625`, which are **`w78:202`'s own leaf `L0[0:2]`** (verified:
`L0 = randn((2,3,3,4,2), float16)` under `manual_seed(12648430)` begins
`[-1.158203125, -0.166015625, 1.306640625, …]`).

`w87:268` cannot produce those numbers from its own data: its only float leaf is
`L0 = randn((1,), float16) = [-0.482421875]` and its other leaf is `uint8`. Furthermore its target
`n9` has shape **`(2,)`** while the archived `dev[:3]` has **three** elements — an impossible
tensor for that output.

**Conclusion.** The user's hypothesis is half right: the identical values *are* a foreign-buffer
signature — but the foreign buffer is **another job's result arriving on the shared DEALER
socket** (B.1), not a device storage-planning overlap. The values belong to `w78:202`, are int8
dequantized, and were mis-attributed to `w87:268`. Both rows are **measurement artifacts**;
neither belongs to a storage-planning class, and neither is graph-opt.
(Caveat: the alternative reading — a genuine stale-arena leak inside the FVP runner across jobs —
is also not a graph-opt bug and is equally disqualifying. Distinguishing them needs the FVP.)

### B.3 (ii) Quantization-boundary saturation — CONFIRMED for both cases the user flagged

For Ethos-U, `build_job(..., quantize=True)` uses `backend_obj.quantized_reference(...)`
(`gen/export/job.py:159-165`), so `j.eager` is the **PT2E-converted** reference, not fp32 eager.
Rebuilt host-side (`eu_probe.py`):

* **`w87:268`** — `n9 = logit(n2)` where `n2 = int32(gt.Scalar(lt(L0,L1), -5))` is all ones, so the
  true value is `logit(1) = +inf`. The quantized reference is `[-128.0, -128.0]` — the **int8 floor**.
  Any device/reference disagreement here is a saturation artifact of representing `±inf` in int8.
  **Confirmed: this is a quantization-boundary case, not a bug.**
* **`w59:999`** — `n6 = ge.Scalar(logit(n0), -2.0)` with `n0 = logical_xor(...) = 0` everywhere, so
  the compared value is `logit(0) = -inf`; reference `= 0.0` (i.e. `-inf < -2`). Device `= 1.0`.
  Under int8 the `-inf` becomes `-128·scale`, which for any `scale < 0.015625` is `> -2` and flips
  the comparison to `1.0`. **The reported device value is the expected consequence of int8
  representability**, not a plan bug.

### B.4 Structural confounds in the remaining rows (all VERIFIED host-side)

Rebuilding the exact A/B graphs `deep_graphopt.py` used:

| row | target | breaking sibling | sibling shape | A `deleg` | B `deleg` | structural problem |
|---|---|---|---|---|---|---|
| `w78:202` out4 | `sigmoid.out` | `n6 = narrow_copy(n4,0,0,0)` | `(0,3,3,4,2)` → **0 elements** | **0** | **0** | empty sibling; **nothing delegated to Ethos-U in either arm**; and eager reference *changes* A→B |
| `w7:1024` out1 | `floor.out` | `n5 = diagonal_copy(...)` | `(2,4,4,0)` → **0 elements** | 3 | 6 | empty sibling |
| `w61:77` out0 | `view_copy` | `n3 = repeat(n0,[3,0,2])` | `(3,0,6)` → **0 elements** | 6 | 9 | empty sibling |
| `w66:845` out1 | `log1p.out` | `n1 = repeat(n0,[3,0,2])` | `(12,0,4)` → **0 elements** | 1 | 2 | empty sibling |
| `w59:999` out3 | `ge.Scalar` | `n2 = bitwise_and.Scalar` | `(1,)` | **22** | **0** | A runs on the NPU, **B falls back entirely to portable** — the A/B compares two different execution paths |
| `w87:268` out2 | `logit` | `n7` | `(3,)` | 7 | 6 | archived dev has 3 elements for a 2-element target (B.2) |
| `w25:933` out2 | `pixel_unshuffle` | `n7 = fmod.Scalar` | `(4,2,2)` | 6 | 6 | none found |

**4 of 7 breaking siblings are zero-element tensors.** Co-returning an empty tensor cannot change
what the NPU computes for the target; it can trivially change output serialization / count /
`cmp.select` alignment. Those four "sibling-dependent" verdicts are confounded at the harness level.

`w78:202` has a second, independent killer: `n4 = sign.out(n1, out=n0)` writes **into `n0`**, and the
target is `n7 = sigmoid(n0)`. The A cone excludes `n4`; the B cone includes it. Measured eager
reference for the *same* output:

```
A[target]      eager[0] = [0.239014, 0.458496, 0.787109, 0.726074, …]
B[target,n6]   eager[0] = [0.5, 0.5, 0.730957, 0.730957, …]      <-- different reference!
```

The "sibling dependence" is the generator's aliased-`out=` mutation (**Claim D's confound**)
changing the oracle, not the backend. And with `deleg=0` in both arms, no Ethos-U graph
optimization is even in play.

`w61:77`'s ZEROED label is also weak: the reference is `[0,0,0,-1,0,0]` and the device is all-zero,
so "dropped store" and "one element wrong" are indistinguishable — the `deep_graphopt.py:89`
ZEROED test (`all |dev|<1e-6 and any |ref|>1e-3`) fires on a single non-zero reference element.

**`w25:933` is the only survivor**: non-empty sibling, genuinely non-zero reference
(`[-1.4308, -0.5014, 1.2719, …]`), device all-zero — a credible dropped store. But even it is
contradicted by its own pinpoint row, which recorded a *non-zero* device value
(`[-1.6877, -0.9050, 0.6604, -0.1223]`) for the same `(job, out)`.

### B.5 Classifier blind spot (source-level)

`deep_graphopt.py` orders its tests `NONFINITE → ZEROED → ALIAS → SCALE → UNSTRUCTURED`, and the
`ALIAS` test compares only against the *one breaking sibling's* eager values. A clobber from any
other tensor (a non-returned intermediate, an input, or another job's buffer) cannot be detected and
falls through to `UNSTRUCTURED`. So `UNSTRUCTURED` is a residual bucket, and the artifact's use of
it as a *mechanism* name is not warranted.

### B.6 Defensible Ethos-U graph-level finding count

All stage counts below were re-counted from the TSVs, not taken from the prose.
`bisect_results.tsv`: `3078 OPERATOR / 470 LOWERFAIL / 284 NOREPRO / 225 GRAPHOPT`.
`graphopt_pinpoint.tsv`: 225 data rows = `163 INCONCL / 36 CONFOUND / 26 GRAPHOPT`, subkinds
`145 TIMEOUT / 36 - / 21 DRIFT / 17 BSKIP / 5 CLOBBER / 1 SKIP`. `graphopt_mechanism.tsv`: 19 rows.

| stage | count | what survives audit |
|---|---:|---|
| raw GRAPHOPT bisection verdicts | 225 | contaminated by B.1; not a finding count |
| pinpoint rows | 225 | **163 INCONCL + 36 CONFOUND** → only **26** were labelled GRAPHOPT (145 of the 225 subkinds are TIMEOUT) |
| mechanism-classified | 19 | **7 of the 26 GRAPHOPT rows were never mechanism-classified at all**; of the 19, **9 are NO_BREAK** (flaky — and B.1 explains the flakiness) → 10 |
| SCALE | 3 | **1** (per `REQUANT_SCALE_REVIEW.md`), and its negative sign is unexplained by "scale" |
| UNSTRUCTURED | 4 | **0** — `w78:202` + `w87:268` are harness mis-attribution (B.2) / aliased-`out=` oracle change; `w59:999` + `w87:268` are int8 saturation (B.3); `w7:1024` has an empty sibling and is a `floor` boundary flip |
| ZEROED | 2 | **1 candidate** (`w25:933`); `w61:77` is a 1-element diff on a near-zero reference |
| NONFINITE | 1 | **0** — empty sibling (`w66:845`) |

**Defensible count: at most 2 Ethos-U graph-level candidates (`w14:133` requant-scale,
`w25:933` dropped store), neither with an archived, reproduced device measurement free of the
B.1 defect.** The honest statement is *"no Ethos-U graph-optimization bug is established"*, with
those two named as needing a re-run on a corrected harness. `145/226 TIMEOUT` and `9/19 NO_BREAK`
should be reported as **coverage failure**, not as evidence of a small-but-real population.

### B.7 Verdict on Claim B

**UNSUPPORTED AS STATED / NEEDS RE-WORDING.** The 7 non-SCALE rows do not support a graph-opt
mechanism: 4 have zero-element breaking siblings, 2 are harness result mis-attribution, 2 are
int8 saturation of `±inf`, 1 has `deleg=0` in both arms plus an aliased-`out=` oracle change, and
1 (`w61:77`) is a single-element diff mislabelled ZEROED. `w25:933` alone remains a candidate.
`GRAPH_CONTEXT.md`'s framing ("dominant mechanism is a requant-scale error, with a secondary
dropped store") should be replaced by a coverage statement plus two named open candidates.
The B.1 harness defect must be fixed and the stage re-run before any Ethos-U graph-level number
is published; it also retroactively weakens the Vulkan A/B in Claim A (2) and the QNN/CoreML/VGF
runs that used the same scripts.

**Not re-run:** anything needing the Corstone-300 FVP. `fvp_client/` exists and looks complete,
but `FVP_Corstone_SSE-300_Ethos-U55` and `arm-none-eabi-gcc` are absent and
`examples/arm/ethos-u-scratch/` is empty, so device re-measurement is **unverifiable now**.
Ethos-U *lowering* works fine here (Vela ran for every case above).

---

## Claim D — portable's 485 shape mismatches are a generator artifact

### D.1 Was it established, or asserted?

Partly established. `findings_v1/portable/mismatch-shape.md` gives 4 fully reproduced cases
(`w0:1246`, `w0:1007`, `w1:288`, `w0:1096`) plus 3 more spot-checks, and quantifies the rest with
**structural proxies** (484/485 contain a `broadcast_to(...,[0,…])`; 476/485 have a degenerate `0`
dim; 289 direct / 196 downstream). The `485/485 = 100% artifact, 0 real` line is therefore an
*inference from a proxy*, not a per-case measurement. I made the per-case measurement.

### D.2 Test 1 — de-alias the `out=` buffer (`portable_shape2.py`)

For all 485 rows: rewrite every `out=nK` / `values=nK` / `indices=nK` / `min=nK` / `min_indices=nK`
/ `max=nK` / `max_indices=nK` to `…=nK.clone()`, so the in-place resize hits a private copy, then
re-run eager (fork-isolated; the corpus heap-corrupts in-process).

```
source contains an aliased out=/values=/min=… node argument : 483 / 485
CONFIRMED (de-aliased eager shape == the shape PORTABLE returned) : 121
DEALIAS_NO_EFFECT (divergence survives de-aliasing)               : 358
DEALIAS_OTHER : 2   NO_ALIASED_OUT : 2   NATIVE_ABORT : 2
```

**This partially refutes the stated mechanism.** For 358/485 (74%) the *aliasing* is not the
operative cause — removing it leaves the divergence intact. That is expected once you look
closely: when the divergent output **is** the multi-output reduction itself (the artifact's
Mechanism A, ~60%), eager returns the correctly-sized result whether or not the buffer is aliased.
Aliasing is only load-bearing for the downstream-inheritance subfamily — my 121 CONFIRMED is the
same population as the artifact's Mechanism B (196), same order of magnitude.

### D.3 Test 2 — is the eager shape independent of the declared buffer? (`portable_shape3.py`)

Perturb every inline `out=torch.empty((…), dtype=D)` to `torch.empty((3,5), dtype=D)` and re-run
eager. If the divergent output shape is unchanged, eager resizes to the op's true shape and simply
ignores the declared buffer, while portable returns the declared shape.

```
EAGER_SHAPE_INVARIANT : 484 / 485      NATIVE_ABORT : 1
```

**484/485 per-case confirmation** that the divergence is exactly "eager resizes a wrong-shaped
preallocated `out=` buffer; ExecuTorch's static planning returns the declared shape". That is a
generator/reference-model artifact and not a portable shape-inference bug — the artifact's
*conclusion* is correct, and now measured per case rather than proxied.

### D.4 Verdict on Claim D

**DEFENSIBLE IN SUBSTANCE; NEEDS A ONE-CLAUSE RE-WORDING.**

* Keep: "0 real backend shape-inference defects; 485/485 are a differential-generator artifact."
  This is now verified per case (484/485; 1 lost to a native abort), stronger than the published
  proxy argument.
* Change: the stated mechanism ("**reusing a node's preallocated `out=` buffer** as a later
  multi-output reduction's output") over-attributes to *aliasing*. The necessary and sufficient
  ingredient is that **the generator preallocates a wrong-shaped `out=` buffer** which eager
  resizes and static planning does not. Aliasing is only required for the ~40% downstream-
  inheritance subfamily (measured: 121/485 change when de-aliased; 358/485 do not).
* Consequence for its use as *the mandatory confound check*: the check must test the **wrong-shaped
  `out=` buffer** condition, not just "is there an aliased buffer". But note the *value*-side
  version of the aliasing confound is real and bites elsewhere: it is the entire content of the
  two XNNPACK GRAPHOPT rows (C.2) and of Ethos-U `w78:202` (B.4). So the confound check should
  cover **both** (a) wrong-shaped preallocated `out=` buffers (shape class) and (b) an earlier
  node's tensor used as a later op's `out=` and then read downstream (value class).

---

## Status of the required edits to `findings/GRAPH_LEVEL_CATEGORY_TABLE.md`

`findings/GRAPH_LEVEL_CATEGORY_TABLE.md` was **rewritten at 10:39**, after this audit was written
(10:24), and the new version already incorporates these results. Verified as applied:

| item | state in the rewritten table |
|---|---|
| Vulkan re-attribution | **applied** — records that forcing the gate open makes the verifier "run and pass" and that `allow_overlapping_allocations=False` works; counts row reads `~7% of jobs / ~3–4% of delegate output slots` against the published `"6 A/B"` |
| XNNPACK zero | **applied** — counts row reads `now *measured*: 325 jobs → 58 OPERATOR / 2 GRAPHOPT, both the generator confound`, and the prose states *"the '0' was previously never measured … Any earlier text calling it 'a result, not a gap' was unsupported"* |
| Ethos-U | **applied** — `1 candidate` vs published `2`, "only `w25:933` survives"; the requant case is handled under the refuted *Numeric plan* category |
| B.1 harness defect | **applied** — listed as methodological defect 7, *"Harnesses discard the returned job id (`deep_graphopt.py:57` et al.)"* |

Still outstanding:

1. **portable's `0`.** The rewritten counts table simply drops the portable row rather than
   re-wording it. `findings_v1/portable/` contains no `graphopt` or `bisect` artifact at all, so if
   a portable graph-opt number is reported anywhere it needs the same "not measured" treatment as
   XNNPACK — or the same in-process re-bisection, which `xnn_graphopt2.py` runs unchanged by
   passing `portable` and `corpus_v1/portable`.
2. **Claim D's aliasing clause** in `findings_v1/portable/mismatch-shape.md` (and the
   "mandatory confound check" wording wherever it is reused). The mechanism sentence still says the
   485 cases come from *reusing a node's preallocated `out=` buffer*; measured, that ingredient is
   not load-bearing for 358/485. See D.2/D.4 for the replacement wording and for the two-part
   confound check (shape class **and** value class).
3. **The `verify_storage_reuse` gate itself** (`memory_planning_pass.py:295`) is a standalone
   ExecuTorch hygiene defect — the check is dead for every backend that passes a
   `MemoryPlanningAlgorithmSuite`, which is the default. Worth reporting upstream independently of
   the Vulkan bug, and worth stating explicitly that it is *not* the cause of that bug.

One framing point survives the rewrite and is worth keeping explicit: the Vulkan result means
"other backends' bugs are not ExecuTorch's shared memory planner" no longer needs to rest on an
XNNPACK null result. `MemoryPlanningPass` is exonerated **directly**, by `verify_storage_reuse`
passing when forced to run (A.1).

---

## Negative results (recorded deliberately)

* Vulkan's delegate-internal memory plan shows **no** storage overlap for the repro graphs, and
  `verify_storage_reuse` **passes** when forced to run — the planner is acquitted. (A.1, A.2)
* `allow_overlapping_allocations=False` is present and effective; nothing about the Vulkan bug
  involves deliberate overlap. (A.2)
* XNNPACK and portable both returned **correct values on every output** of 8 and 10
  aliasing-shaped multi-output graphs — no dropped store found by direct probing. (C.2a)
* De-aliasing the `out=` buffers does **not** repair 358 of 485 portable shape mismatches — the
  published mechanism's aliasing ingredient is not load-bearing for the majority. (D.2)
* No device re-measurement was possible for Vulkan (no phone) or Ethos-U (no FVP, no
  arm-none-eabi toolchain). Every device number in Claims A and B remains **unverifiable now**.

---

## Summary

**Two results dominate.**

*Vulkan (A).* The write-up's mechanism is wrong; the real one is better. The
`verify_storage_reuse` gate is genuinely closed — `_callable_name(MemoryPlanningAlgorithmSuite())`
is a repr, not `"greedy"` — and forcing it open makes the verifier run and **pass** on every repro
graph. The delegate plan has no storage overlap at all, so the planner is acquitted. The actual
fault is an output-arity shortfall: `RemoveRedundantOpsTransform` deletes the `clone`, both graph
outputs collapse onto `n0`, `process_output_node` iterates the *deduplicated* `all_input_nodes`
(`vulkan_graph_builder.py:442`) so the blob declares 1 output for 2 call-site slots, and
`VulkanBackend.cpp:821` (`output_offset = args.size() - num_outputs`) leaves the leading slot
**unwritten**. Emptying `redundant_ops` removes the shortfall (causal test), and
`prod(acos(zeros)).half() = 2.466796875` reproduces the reported device value `2.4668` exactly.
It fires for `clone`, `alias_copy`, and shape-preserving `expand_copy`, only when two graph-output
edges collapse onto one value — which is why the corruption looked output-set-sensitive. Vulkan
therefore belongs with the VGF/CoreML *dropped-store* class, not the memory-planner class.

*XNNPACK (C).* The zero was never measured: no `bisect_results.tsv`, no GRAPHOPT token, no
`BISECT_BACKEND=xnnpack` anywhere, and the bisection tooling postdates the run by 2–5 days while
11,024 unbisected MISMATCH rows sat ready in the format the bisector consumes. "A result, not a
gap" is unsupported. I ran the missing experiment in-process: the graph-opt bucket is small but
**non-empty**, and every member is the generator's aliased-`out=` confound — no storage-planning
bug, and 8/8 hand-built aliasing graphs returned correct values. XNNPACK's serializer also
*rejects* the partition shape that silently drops a store in Vulkan. The conclusion survives with
better evidence, but every sentence resting on the unmeasured `0` (portable's included) needs
re-wording.

*Ethos-U (B).* Not defensible. Every harness script discards the returned job id
(`deep_graphopt.py:57` et al.), so with 145/225 TIMEOUTs late results land on the wrong job —
which is why `w78:202` and `w87:268` carry byte-identical device values that are provably
`w78:202`'s own int8-dequantized input and cannot arise from `w87:268`'s data. 4 of 7 breaking
siblings are zero-element tensors; two rows are int8 saturation of `±inf`; `w78:202` has
`deleg=0` in both arms plus an aliased-`out=` oracle change. Only `w25:933` remains a candidate.
Report coverage failure, not a small real population.

*Portable (D).* Substance holds and is stronger: 484/485 per-case confirmation that eager's
divergent shape is independent of the declared `out=` buffer. One clause needs fixing — *aliasing*
is not the operative ingredient for 74% of cases; a **wrong-shaped preallocated `out=` buffer** is.
