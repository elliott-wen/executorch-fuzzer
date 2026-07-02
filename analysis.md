# Analysis workflow — backend fuzzing & bug localization

Backend-agnostic process for finding and localizing bugs in an ExecuTorch backend (Vulkan,
XNNPACK, QNN, Core ML, …) by feeding a corpus of random graphs to a real device and triaging
every non-OK outcome into a confirmed, device-verified bug.

**Core rule: every claim is verified by running on the device. Nothing is concluded from op names
or statistics alone** — enrichment tables only point at suspects; the device decides.

**No borrowed verdicts.** A finding for *this* device must come from bisecting *this* device's own
failing graphs. A prior run, another device's results, or a high cross-device concordance number
may *motivate* what to expect, but is NEVER a substitute for re-bisecting on the device under test.
The cross-device differential (optional section) is an *additional* product computed from two
independent full bisections — not a shortcut that lets you skip one of them.

## The two axes of a finding

Each fed graph returns one of five outcomes: **OK / MISMATCH / CRASH / SKIP / TIMEOUT**. Every
non-OK outcome is triaged along two independent axes.

**Axis A — failure mode (what the device did). The report files a bug list per mode:**
1. **MISMATCH bug** — the graph ran but produced numerically wrong output (wrong values, dtype,
   shape, or nan/inf positions vs the eager CPU reference).
2. **CRASH bug** — the device runtime aborted (native crash) loading or running the `.pte`. A
   reproducible **TIMEOUT** (survives re-run on a healthy worker) is a hang — triage it as a CRASH.
3. **SKIP bug** — the backend lowered the graph (it's in the corpus) but the device **rejects it at
   runtime** — an op/pattern that passes partitioning yet can't actually execute. These are real
   coverage gaps. (Distinguish from a feeder-side SKIP — a non-tensor output that can't be compared
   — which is not a backend bug.)

**Axis B — root cause (why), which applies to MISMATCH and CRASH alike:**
1. **Operator bug** — a single op's kernel computes the wrong result / crashes on its own.
2. **Graph-optimization bug** — every op is correct in isolation, but a graph-level pass (memory
   planning, buffer aliasing / copy-elision, fusion, dtype/layout rewriting) corrupts the result —
   or crashes — only when the op runs in a larger graph. The required extra nodes are the trigger.

The whole point of the workflow is to localize each non-OK graph to one cell of this grid, with a
minimal device-verified repro.

## 0. Setup
- A corpus of random graphs lowered to the target backend, found in `corpus_v2/<backend>`.
- The broker is up locally and a worker (device or emulator) is connected and pulling — **a human
  sets this up**. Your job starts at the feed step below.

## 1. Feed the corpus → collect every non-OK outcome
```
python -m mobile feed --corpus <corpus> --skip-log <skiplog.tsv>
```
The feeder holds the eager reference, runs each graph on the device, diffs the outputs, and logs
every non-OK graph (`status, job_id, reason, op_chain`). Each MISMATCH names the first failing
output `out[N]` — the divergence to localize. Write to a **fresh** `--skip-log` path; **never
overwrite an existing log**.

Then split the log into the three finding buckets up front — `MISMATCH`, `CRASH`, and
worker-reported `SKIP` — and work each through steps 2–5.

### Fan out — shard across processes, keep full coverage
Steps 2–4 are embarrassingly parallel (each failing graph localizes independently), and the
bug-localization modules (`gen/diff`) do the on-device runs via the injected predicate. Parallelize by
**sharding the worklist across processes**, not by single-threading one big run:
- **Shard**: launch N bisector processes, each taking `work[i]` where `i % N == shard_id`, each with
  its **own** results file and persistent dealer, all pointed at the **one** broker. Merge at the end
  and assert `rows == MISMATCH count` (the coverage gate).
- **Size N by idle cores, not the device.** The device/broker is the shared serialization point but
  isn't the bottleneck — each graph is mostly host-side `build_job` and only ~1–2 s on-device. The
  only reliable knob is **N**: `OMP_NUM_THREADS`/`torch.set_num_threads` do NOT cap `build_job` (torch
  export sizes its own pools to `nproc`), so on a free box N shards oversubscribe ~N× and thrash. Tune
  N against `cat /proc/loadavg` (aim load ≈ cores); the box is multi-tenant (pregen fleets come and
  go), so re-check `nproc`/`load` at launch and run the sweep as a long **resumable** background job.
- **Detach via the harness background mechanism** (or `setsid`), not `nohup … &` inside an agent shell
  (killed when the call returns); verify `pgrep` shows the shards alive a minute later.
- **Bounded device traffic bounds concurrency, not coverage.** Keep each shard's in-flight count small,
  but the full MISMATCH set MUST be covered — slow is acceptable, silent under-coverage is not.
- **Diagnose "slow" before reacting.** A wave of `LOWERFAIL`/`dev=TIMEOUT` is usually a **dropped
  device worker**, not a code bug — confirm liveness with a one-job feed (`python feed.py <job_id>
  --corpus … --timeout 30`) before touching anything; the resumable shards resume where they left off.

## 2. Bisect each failing graph → minimal reproducing subgraph
**Treat each failing job independently — bisect it on its own before any aggregation.** Do NOT
roll the skip-log up into op-frequency rankings and reason from those: co-occurrence over-blames
(a frequent op looks guilty just for riding along in graphs that fail for another reason). Take
each failing `job_id` from the log, minimize it, and let the device decide — one repro at a time.
Aggregation (op enrichment, culprit tables) is **provenance only, produced AFTER** per-job
bisection to organize confirmed findings, never as the basis for a verdict. ** Never shortcut, always bisect each case in any situations **

**Coverage gate — the definition of done for step 2.** Step 2 is complete only when **every
MISMATCH `job_id` in the skip-log has its own verdict row** in a `bisect_results.tsv`
(`job_id, out_idx, verdict, cone_size, required_siblings`). The number of bisected jobs MUST equal
the MISMATCH count — assert it. Bisection is a **per-job** obligation, not a per-op one: N distinct
corpus graphs whose diverging output is produced by the same op are N separate bisections, because
each can minimize to a different cone, a different sibling trigger, or turn out flaky. A
"representative example per op" is an *illustration* of a confirmed verdict, never a stand-in for
the missing rows. If you genuinely cannot finish in the time available, that is a **partial**
result: report exact coverage ("bisected X of N mismatches"), list the un-bisected `job_id`s, and
never let enrichment tables or a prior run imply the gap is covered (no silent caps).
(CRASH and SKIP are exempt from per-output bisection — by their nature they are localized by
single-op isolation in steps 3–4, not output-set ddmin — but each must still be device-verified.)

**Use `mobile.gen.diff` — do NOT hand-roll a bisector.** The delta-debugging localizer is a reusable,
tested module; you supply only a device predicate, and the two-axis minimization is built in.
- `gen/diff/ddmin.py` — `ddmin(items, still_fails)`, the pure primitive (git-bisect over a *set*):
  the smallest subset that still fails. No torch/broker.
- `gen/diff/bisect.py` — the two localizers on top of it (each takes an injected predicate, returns a
  verdict, and re-checks the full set/cone first — returning **INCONCLUSIVE** if it doesn't reproduce):
  - **Axis 1 — returned outputs:** `bisect_output_set(target, others, target_diverges)` →
    **CONE_LOCAL** (wrong with no sibling outputs) / **SIBLING_DEPENDENT** (graph-opt triggered by a
    co-returned output — those outputs are the trigger) / INCONCLUSIVE.
  - **Axis 2 — the input cone:** `bisect_cone(target, ancestors, diverges_with_live)` → **OPERATOR**
    (wrong even with the target computed alone on baked-correct inputs → kernel bug) / **COMPOSITIONAL**
    (graph-opt along the path — wrong only with specific upstream ops run live) / INCONCLUSIVE.
- `gen/diff/cone_predicate.py` — `cone_localizer(graph_path, out_idx)` → `(ancestors, build_src,
  target)`: parses the graph with `ast`, runs the eager reference once, and emits the baked sub-graph
  source for any kept-ancestor set (every removed ancestor baked to its eager value). Backend-agnostic.
- `tmp/cone_localize.py` — the reference driver wiring the predicate to the broker. Copy it.

A full localization is `bisect_output_set` → if CONE_LOCAL, `bisect_cone`. **CONE_LOCAL is NOT
"operator"** — the target's whole chain still runs, so it can be a single-op bug *or* a graph-opt
along that chain; only `bisect_cone` splits it. You implement ONE injected function:
`run_and_diverges(src) -> bool` = `build_job(src, backend, quantize)` → broker → `compare()` the
target → True iff it diverges. The ddmin search, the faithful reference-baking, and the
operator/graph-opt split are done for you.

**The predicate still owns the oracle + env contract:**
- **Exact oracle** — use `net/compare.py` (`compare()`/`_cmp`): per-dtype rtol/atol, nan/±inf
  positions element-wise. If you index a single output, first drop ExecuTorch's PREPENDED
  input-mutation outputs — `if len(et) > len(eager): et = et[len(et)-len(eager):]` — before `et[T]`;
  skipping this silently flips verdicts on any `out=`/mutation graph.
- **Exact inputs** — `cone_predicate` bakes cut inputs from the graph's own eager intermediates (same
  `LEAVES`), so data-dependent bugs (`floor_divide(x,x)→0`, fp16 overflow, OOB index) survive. Don't
  substitute fresh random inputs.
- **Right backend + env** — `build_job(src, <backend>)` under that backend's env (next box), else it
  SKIPs everything and reads as "bug gone".
- **Stability** — the localizers already re-check the full set/cone before splitting (→ INCONCLUSIVE
  if flaky); still gate every *confirmed* operator bug with **N≥5 re-runs** (int8 near-tolerance is
  non-deterministic — see the gate lesson).
- **CRASH/SKIP predicate** — the same injection works for non-mismatch modes: make `run_and_diverges`
  return True on `status==CRASH`/abort or on worker `SKIP`. Tolerate a worker respawn mid-run
  (re-send the candidate) rather than logging a false "didn't reproduce".

**Resumable coverage.** One results row per `(job, out)`; treat a row as done ONLY if it has a real
verdict — re-attempt every `LOWERFAIL`/`ERROR`/`dev=TIMEOUT` (during a device outage all in-flight
graphs log TIMEOUT; counting those as done silently drops thousands and the coverage gate passes on a
lie). Record *why* a run failed (`build=<status>` vs `dev=<status>`), sanitize the `detail` field
(strip tabs/newlines), and loop resumable sweeps until every MISMATCH has a real verdict.

> **Re-lowering a graph to a `.pte`** — if you're unsure how to lower a (sub)graph source to the
> target backend, `pregen.py` is the working reference: it lowers graphs to every supported backend
> via `build_job(src, backend)` (in `gen/export/job.py`) and serializes the result. Copy that call —
> same `build_job` the corpus was generated with — for rebuilding bisect candidates and repros.
>
> **Re-lower in the SAME environment the corpus was generated in** — some backends need special
> env/interpreter or `build_job` silently SKIPs (or the backend never registers). The launcher
> scripts are the canonical setup; mirror them:
> - **QNN/Qualcomm** (`run_pregen_qnn.sh`): `source android-dev/android-env.sh` first — it exports
>   `QNN_SDK_ROOT` and puts the host `libc++`/libunwind + QNN libs on `LD_LIBRARY_PATH` (RHEL 9
>   lacks them). Also needs `flatc` on `PATH` (it's in `.venv/bin`); without it every QNN lowering
>   fails. Runs in the 3.12 `.venv`.
> - **MediaTek** (`run_pregen_mtk.sh`): runs in the **Python 3.10** `.venv-mtk` (mtk_converter is
>   cp310-only); do NOT source android-env there. See `neuropilot_sdk/README-venv-mtk.md`.
> - **Vulkan / XNNPACK / portable**: the 3.12 `.venv` with `flatc` on `PATH`; no extra SDK env.
>
> Sanity-check the backend registered before bisecting: `build_job` returns a SKIP reading
> "unknown backend … (have: …)" when the env is wrong.

## 3. Classify: operator vs graph-optimization
Read the verdict off the two localizers (both axes), never off the output set alone. **CONE_LOCAL is
not a classification** — it only means "no sibling output needed"; you MUST run `bisect_cone` to split
it:
- `bisect_cone` → **OPERATOR** — the producer, computed alone on its exact baked-eager inputs, still
  fails as the only output → the kernel is wrong / crashes on its own. Example: `floor_divide.default`
  (`w14:141`) isolates to the one op.
- `bisect_cone` → **COMPOSITIONAL**, or `bisect_output_set` → **SIBLING_DEPENDENT** — a
  **graph-optimization bug**: the producer is correct alone but fails once specific upstream ops run
  live (COMPOSITIONAL, `needed_live`) or once a specific sibling output is co-returned
  (SIBLING_DEPENDENT); those nodes/outputs are the trigger. Example: `copy.default` (`w16:995`) is
  clean alone, COMPOSITIONAL only with `n3,n4` live — a Vela number-format choice, not a copy kernel.
- Either localizer → **INCONCLUSIVE** — the failure didn't hold up under re-check. **Do NOT just drop
  it — investigate in detail**, because INCONCLUSIVE lumps three very different things and one of them
  is a real, high-value bug. Determine which by re-running the **full** failing graph N≥5×:
  1. **Full graph reproduces every time, but dies on *any* reduction → whole-graph compilation bug**
     (real — file it). The full-graph calibration / memory plan / fusion *is* the trigger, so no
     subgraph reproduces it and reduction structurally cannot localize it. Localize instead by
     inspecting the compiler's decisions on the full graph: dump the target tensor's
     `(scale, zero_point)`, clip range, and fusion/memory plan, and diff against the correct
     quantization. File as a graph-optimization bug with that compiler artifact as the evidence.
  2. **Full graph itself doesn't reproduce N× → genuinely flaky / near-tolerance** — record as flaky,
     don't file.
  3. **Target won't lower alone (`BUILD:SKIP`, strict partitioning) → can't-isolate** — use the
     constant-output probe (below) or the full-graph inspection of (1); never leave it at "flaky".

**A graph-opt verdict is only half a finding — name the MECHANISM (the "why"), don't stop at the
trigger.** For every SIBLING_DEPENDENT or COMPOSITIONAL case, `needed_live`/the trigger siblings tell
you *which* nodes matter, not *why* co-computing them corrupts the target. Characterize the target's
**wrong device value** against the reference and the trigger's values:
- **ALIAS / clobber** — `dev_target ≈ a sibling's (or another live tensor's) value`, verbatim → the
  memory planner aliased the target's output buffer onto a still-live buffer. Confirm by matching the
  wrong value against every graph tensor (clobber-source matching) to **name the aliased buffer**;
  neutralize the correct buffer — sometimes the aliasing is on a shared *input*, so copy the input
  fresh, not just the output (a `+0.0` on the output that "doesn't fix it" means you neutralized the
  wrong buffer).
- **SCALE:r** — `dev_target ≈ reference × r` (a consistent factor, incl. sign flip or wild value) → a
  **requantization-scale error**: the trigger changed the compiler's quant plan for the target.
- **ZEROED** — `dev_target` all-zero → a **dropped store** (the output isn't written under the
  multi-output plan).
- **NONFINITE** — `dev_target` nan/inf where the reference is finite → overflow the plan introduced.

These mechanisms differ per backend (QNN = buffer aliasing; Ethos-U = requant-scale) — derive it from
*this* backend's device values, don't assume it transfers. File the bug with the named mechanism +
the device-verified `wrong-value ↔ source` evidence, not just "diverges with sibling X".

Confirm an optimization verdict with an **A/B neutralization**: replace the suspected
optimization-triggering op with an equivalent that forbids the optimization (e.g. swap a
value-preserving copy for a fresh-buffer compute — `add 0.0` into a new buffer), rebuild, re-run.
If the failure vanishes → it was the optimization, not a kernel (see
`bugs/vulkan-copy-elision-aliasing.md`). The same split applies to CRASH: an op that crashes only
in a larger graph but runs correctly alone is a graph-optimization confound (the Vulkan copy/view
crash cluster), not an operator crash.

**On strict-partitioning backends, "diverges only with a sibling" is often a LOWERING artifact.** If
the target won't partition alone (`build_job` → SKIP, e.g. QNN won't lower a lone
`fill.Scalar`/`copy`/`roll`), `bisect_cone`'s target-alone check can't run and the case is
**can't-isolate → inconclusive**, NOT a graph-opt bug — map a `BUILD:SKIP` in your predicate to that,
don't let it read as "no reproduction" and keep the sibling. `bisect_cone`/`bisect_output_set` already
re-check the full set/cone and return INCONCLUSIVE for flaky cases; treat any GRAPHOPT/SIBLING_DEPENDENT
that never confirms as a candidate, never a finding.

**When the op won't isolate, use a CONSTANT-OUTPUT probe to confirm graph-opt anyway.** The reason
the A/B test needs "lower alone" is to get a trusted oracle for "correct." You can skip that
entirely by making the **target an op whose correct answer is known from semantics**, independent of
the rest of the graph: `fill.Scalar(x, C)` / `zeros` / `ones` / `full` always output the constant.
Then in a graph returning `{fill, sibling}`, if the device's fill output is **not all-C** (and the
sibling is data-independent of the fill — not in its cone), the constant was overwritten → a
**buffer-aliasing / memory-planning** graph-opt bug, proven without ever isolating the op. This is
how the QNN graph-opt bug was found (`tmp/qnn_fill_probe.py`): `fill.Scalar(·, 7)` returned
`[-0.482, 1.11]` on device, stably — the memory planner aliased the constant-fill buffer with a live
tensor. (`fill` is the #1 QNN GRAPHOPT target, 520 cases.) **Caveat from the ASUS run:** forcing a
fresh buffer on the target's *output* (`+0.0`) did NOT neutralize an `expand_copy` case — the
aliasing was on the shared *input*, not the output. So when neutralizing, copy the **input** fresh,
not just the output; "the +0.0 didn't fix it" means you neutralized the wrong buffer, not "not a
graph-opt bug."

**Filter non-finite first — it's usually propagation, not a sink-op bug.** If the diverging output
is nan/inf, an upstream op typically produced the non-finite value and it merely flowed down (the
oracle already treats *matching* nan/inf positions as OK, so only *differing* positions flagged).
Trace to the first op that produces the non-finite value and attribute there, or mark it as
propagation and do NOT file the sink op. (This is why whole-graph enrichment over-blames `softmax` /
`mean` / `pow` — their corpus "fails" are mostly nan/inf passing through.)

SKIP is its own classification: confirm the rejection is real by running the op as a **one-op
graph** on the device. If it SKIPs in every form → genuine unsupported-op coverage gap (file it,
enumerating the rejected forms). If the one-op graph runs → the skip was triggered by the
surrounding graph (a partition/lowering interaction), and the trigger nodes are the finding.

## 4. Verify root causes on the device too
A mechanism hypothesis is not a finding until tested on the device. Test the proposed cause
directly and let the device confirm or refute it (e.g. if you suspect "x/x rounds below 1", run
`div(x,x)` on the device; if you suspect an op is unsupported, run it as a one-op graph and see if
it SKIPs). State only device-verified input→output pairs; mark anything unpinned as not pinned.

## 5. Write each bug with a self-contained repro
One `bugs/<bug>.md` + `bugs/repro_<bug>.py` per confirmed bug. The repro builds the `.pte`, runs it
on the device via the broker, and prints eager vs device (for a SKIP/CRASH, prints the device
status). Label each bug with **both axes**: its failure mode (mismatch / crash / skip) and, for
mismatch/crash, its root cause (operator / graph-optimization). List ops investigated but **passed
in isolation** as NOT filed, so the report separates confirmed bugs from ruled-out suspects (a
crash/mismatch that is correct alone is a graph-optimization confound, not an operator bug).

## Optional: cross-device differential (when ≥2 devices ran the same corpus)
Same corpus → diff per job. A confusion matrix separates **shared** failures (backend-level, every
device) from **device-unique** ones (driver/precision-specific). Bisect the device-unique ones the
same way (steps 2–3). Do this per failure mode — shared vs unique mismatches, crashes, and skips.

## Outputs (this folder, per device)
- `README.md` — synthesis, with a **section per failure mode**:
  - **Mismatch bugs** — correctness divergences, each labeled operator or graph-optimization.
  - **Crash bugs** — native aborts (and reproducible hangs), each labeled operator or
    graph-optimization; call out the graph-opt confound cluster explicitly.
  - **Skip bugs** — ops/patterns the backend lowers but the device rejects at runtime (coverage
    gaps), with the rejected forms enumerated.
- `bugs/` — each confirmed bug (md + runnable repro), grouped/labeled by failure mode and root
  cause; `bugs/INDEX.md` lists them.
- Supporting tables (per-output attribution, crash/skip enrichment, cross-device differential) as
  needed — provenance only; every entry that becomes a finding is device-verified.

## Lesson: don't design expensive isolation around an unverified premise (Ethos-U FVP builds)
The Ethos-U per-`.pte` runner build first looked like it needed a private 251MB SDK copy per job
("the build mutates the SDK in-source → concurrent builds corrupt each other"). That premise was
**never verified and was false**: a build touches **0 files** under the SDK
(`find SDK -newermt '-3 min'` → empty). The real costs were two unrelated things, both fixable
without copying anything:
1. **A full gitlab manifest RE-SYNC on every build.** `examples/arm/executor_runner/CMakeLists.txt`
   calls `arm_ensure_ethos_u_content()` gated by `FETCH_ETHOS_U_CONTENT`, whose default
   (`arm_ethos_u_default_fetch`) turns **ON** unless `core_platform` AND `core_software` already
   exist at `ETHOS_SDK_PATH`. A slightly-incomplete private copy ⇒ re-clones the whole SDK
   (core_platform, core_software, openamp, freertos, threadx, vela…) from gitlab.arm.com **per
   build** → ~2 min/build, network-bound, and the thing that actually raced. Fix: point at a
   **complete pre-synced SDK** and pass **`-DFETCH_ETHOS_U_CONTENT=OFF`** (the SDK is already there).
2. **A cmsis-nn duplicate-target.** In a combined Ethos-U+Cortex-M runner the Ethos-U driver
   (core_software) creates the `cmsis-nn` target, then `backends/cortex_m/CMakeLists.txt` adds it
   again unguarded → "target cmsis-nn already exists", configure fails. Fix: guard with
   `if(TARGET cmsis-nn) … reuse … elseif(CMSIS_NN_LOCAL_PATH) …`.

Result: **shared read-only SDK** (one copy, all clients) + per-job **OUTPUT** dir only (deleted by a
trap) + `ARM_SKIP_PATCH=1` → **~8s/build**, no copy, no fetch, no race, bounded disk (~330MB ×
in-flight only). Adaptive per-`.pte` op linking (`--select_ops_list` from the graph's ops) is kept.
**General rule: before building isolation/caching machinery to dodge a "corruption/race", first
prove the corruption exists** (diff the shared resource across a run). Here the unverified premise
cost a 702GB disk blowup and a `/tmp` ENOSPC outage; a one-line `find -newermt` would have refuted it.

## Lesson: on int8/low-precision, single-pass verdicts over-report ~5× — gate everything
Two rigor failures on the Ethos-U run, both caught only by re-testing (write these into every run):
1. **A single GENUINE isolation is NOT a confirmed operator bug.** Re-running each of 26 first-pass
   GENUINE ops **5×** collapsed them to **5 STABLE** (5/5), 7 intermittent, **10 flaky (0/5 — never
   reproduced)**. Near-tolerance int8 divergences are non-deterministic; one run is noise. Gate every
   candidate with N≥5 re-runs; keep only all-N reproducers as bugs, report ≥1/N as INTERMITTENT, drop
   0/N. Skipping this over-reports operator bugs several-fold. (Vulkan did 7×; don't skip it.)
2. **"dtype divergence" (device int vs reference float) is usually an isolation artifact** — a one-op
   quantized graph can legitimately keep a quantized-int output. Both dtype survivors were 1/5. Treat
   dtype-only isolation mismatches as suspect until reproduced.
3. **Filter reference-side confounds before claiming a device bug.** Huge `2^63` deltas were INT64
   sentinels in the CPU *reference* (cumsum-type int64 ops on the CPU fallback), not device garbage —
   they mismatch *alone*. On a "diverges only with sibling" (GRAPHOPT) claim, run the target ALONE
   first: if it mismatches alone it's CONFOUND, not graph-opt. 57% of Ethos-U GRAPHOPT verdicts were
   confounds. To name the real graph-opt mechanism, characterize the wrong device value: `ALIAS`
   (==a sibling's buffer → memory aliasing), `SCALE:r` (==ref×r → requant-scale bug), `ZEROED`
   (dropped store), or non-finite. Don't assume the QNN mechanism transfers — Ethos-U's is a Vela
   requant-scale error, QNN's was buffer aliasing.

Two harness gotchas this surfaced: (a) **legitimate SKIPs** dominate parts of the Ethos-U corpus —
e.g. `aten::_fft_r2c.out` has **no portable kernel**, so the FVP load fails with "Missing operator"
(`exit 2` → SKIP). That's a real coverage gap of the *target*, not a build/SELECT_OPS bug — enumerate
these in the Skip-bugs section, don't chase them as failures. (b) **Long-lived helper processes**
(broker, FVP clients) only survive when launched as a **bare single-line `exec <prog>`** via the
harness background mechanism; a multi-line command (with `source …`/`set -x`, or a supervisor that
`wait`s) gets torn down when the launching tool-call shell exits. One process per background task.
