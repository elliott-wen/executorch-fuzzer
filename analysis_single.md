# Analysis workflow (single-operator corpus) — per-operator bug isolation

A process for finding and characterizing bugs in an ExecuTorch backend (Vulkan, XNNPACK, QNN,
Core ML, …) using a **single-operator corpus**: graphs generated with `--nodes 1`, where each graph
is exactly one operator wired directly to leaf inputs — no adapter nodes, no other operators, one
real output. You feed this corpus to a real device, and every non-OK outcome is already a minimal
repro attributed to exactly one operator. Your job is to confirm each one on the device, filter the
confounds, and name the reason it fails.

**Core rule: every claim is verified by running on the device.** Nothing is concluded from op names,
decomposition, or statistics alone. The per-operator table points at suspects; the device decides.

## What this catches — and what it does not

Because every graph is a single operator, there is nothing to bisect and no co-occurrence between
ops. Each failing graph *is* the localized finding. This gives you, per operator and across many
shapes/dtypes/values:

- **Operator kernel bugs** — the op computes the wrong value / dtype / shape, or produces nan/inf
  where the reference is finite.
- **Operator crashes** — the kernel aborts loading or running the op alone.
- **Unsupported-op skips** — the op lowers into the corpus but the device rejects it at runtime (a
  real coverage gap), enumerated per rejected form.
- **Intra-op lowering bugs** — a high-level op that decomposes into several edge ops, or is
  requantized during its own lowering, and is wrong because of that op's lowering.

What a single-operator corpus **cannot** surface: bugs that only appear when an op runs *alongside
other operators* — memory planning, buffer aliasing / copy-elision, cross-op fusion, dtype/layout
rewriting across nodes. A one-op graph has no siblings and no multi-node memory plan, so these never
trigger. Do not go looking for them here; every finding in this workflow is attributable to a single
operator (including its own lowering).

## The one axis: failure mode

Each fed graph returns **OK / MISMATCH / CRASH / SKIP / TIMEOUT**. You triage every non-OK outcome
into one of three finding buckets:
1. **MISMATCH** — the op ran but produced numerically wrong output (wrong values, dtype, shape, or
   nan/inf positions vs the eager CPU reference).
2. **CRASH** — the device runtime aborted (native crash) loading or running the op's `.pte`. A
   reproducible **TIMEOUT** (survives re-run on a healthy worker) is a hang — triage it as a CRASH.
3. **SKIP** — the backend lowered the op (it's in the corpus) but the device **rejects it at
   runtime**: an op/form that passes partitioning yet can't execute. A real coverage gap.
   (Distinguish from a *feeder-side* SKIP — a non-tensor output that can't be compared — which is not
   a backend bug; drop those.)

Every non-OK outcome is attributable to one operator by construction. But "isolated to one op" is not
the same as "confirmed backend operator bug" — several confounds masquerade as one and MUST be
filtered (step 3). That filtering is the real work.

## 0. Setup
- A **single-operator** corpus lowered to the target backend, generated with `--nodes 1`
  (`pregen_fleet.py --nodes 1 --backend <backend> --total <k·N>`; size `--total` so each of the
  ~228 usable ops is sampled `k` times — see the coverage note). It lives in `corpus_v2/<backend>`
  (or wherever the run wrote it).
- Each job carries a **delegation breakdown** — a first-class triage signal (step 3a):
  - in the `.py` header comment: `# delegated: backend=<b> ops=<d> non_delegated=<m> delegate_calls=<c>`
  - in the `.job` header JSON: `"delegated": {"ops": d, "non": m, "calls": c}`
  - `ops` = real ops the backend delegate absorbed for this graph; `ops=0` means the op fell back to
    portable CPU kernels (the delegate never ran it).
- The broker is up locally and a worker (device or emulator) is connected and pulling — **a human
  sets this up**. Your job starts at the feed step.

## 1. Feed the corpus → collect every non-OK outcome, keyed by operator
```
python -m mobile feed --corpus <corpus> --skip-log <skiplog.tsv>
```
The feeder holds the eager reference, runs each graph on the device, diffs the outputs, and logs
every non-OK graph (`status, job_id, reason, op_chain`). Write to a **fresh** `--skip-log` path;
**never overwrite an existing log**.

Every graph is one op, so the `op_chain` field *is* the operator under test — no chain to parse.
Split the log into the three buckets (`MISMATCH`, `CRASH`, worker `SKIP`) and group each by operator.
From here you work per-operator.

## 2. Build the per-operator table (the worklist and coverage ledger)
Join the skip-log with the delegation breakdown into one table — one row per **(operator, form)**,
where a form is the signature that varies across the op's `k` samples, typically
`(dtype, rank/shape-class, delegated?)`:

| operator | samples k | OK | MISMATCH | CRASH | SKIP | delegated (ops≥1) | portable (ops=0) |
|----------|-----------|----|----|----|----|----|----|

This table is both your worklist and your coverage ledger. Because there is no co-occurrence between
ops, the table is a sound basis for verdicts (not just provenance): each row is one operator, and
each becomes a device-verified finding or a ruled-out suspect.

**Coverage gate — the definition of done.** The seed schedule samples every op `k ≈ total/N` times
(N ≈ 228 usable ops), so completeness is *per operator*:
1. Every operator that appears in the corpus has a verdict row (OK, confirmed bug, or filtered
   suspect).
2. Every operator **scheduled but absent** from the corpus (never lowered despite equal scheduling —
   seed-infeasible, or a `build_job` SKIP/crash at generation time) is listed with its attrition
   reason. That is a coverage gap of the generator/backend pair, distinct from a device bug.
3. Assert `covered_ops + gap_ops == scheduled_ops`; report exact coverage ("verdicts for X of N
   scheduled ops; Y ops never lowered"). No silent caps.

## 3. Gate and filter the confounds (the real work)
Before calling any per-op divergence a backend operator bug, clear it through **every** filter below —
each one masquerades as a single-op operator bug.

**3a. Bucket by delegated-vs-portable FIRST — this decides whose kernel ran.** Read `delegated.ops`
from the job header:
- **`ops ≥ 1` (op ran on the delegate):** a MISMATCH/CRASH is the **backend's delegated kernel** for
  this op. This is the finding you want.
- **`ops = 0` (op fell back to portable):** the delegate **never ran this op** — the device executed
  the portable CPU kernel. A divergence is then a **portable-kernel or reference issue, NOT a bug in
  the backend under test.** Do not file it against the backend (note it separately if it looks like a
  real portable bug). Apply this before anything else — it removes a whole class of false
  attributions.

**3b. Determinism gate — N≥5 re-runs of the SAME job.** Keep two notions of repetition separate:
- **Input-diversity (the `k` corpus samples):** the op's `k` scheduled samples use *different*
  shapes/dtypes/values (generation is stochastic). "Fails in m/k samples" measures robustness across
  inputs — good for finding form-specific bugs, but it is NOT a determinism check.
- **Determinism (re-run one instance):** pick a failing `job_id` and re-run that **exact** `.pte`
  **N≥5×**. int8 / low-precision near-tolerance divergences are non-deterministic; a single pass
  over-reports operator bugs several-fold. Keep only all-N reproducers as bugs; report ≥1/N as
  INTERMITTENT; drop 0/N (flaky, don't file).

**3c. dtype-only mismatch is usually an isolation artifact.** A one-op quantized graph can
legitimately keep a quantized-**int** output where the fp32 reference is float — and with a single-op
quantized corpus this case is everywhere. Treat any device-int-vs-reference-float mismatch as
**suspect** until reproduced as a genuine *value* error (compared in a common space), not merely a
dtype difference.

**3d. Reference-side confounds — sanity-check the reference.** Some ops mismatch on *every* sample
because the **CPU reference itself** is degenerate, not the device: INT64 sentinels / `2^63` deltas
from int64 CPU-fallback ops (cumsum-type), or data-dependent reference quirks. Before filing, confirm
the eager reference is itself sane for that op/dtype; a reference artifact is a confound, not a device
operator bug.

**3e. Non-finite — check the input leaf.** In a single-op graph there is no upstream op, so nan/inf
where the reference is finite **is the op itself** (fp16 overflow, precision loss) — a genuine
finding. The only exception: verify the **leaf input** wasn't already non-finite in the reference run
(the oracle treats *matching* nan/inf positions as OK, so only *differing* positions are flagged; a
leaf that's finite but the op still emits nan/inf on device → real op bug).

## 4. Name the reason per operator
For each operator that survives step 3, characterize the wrong device value against the reference and
give the mechanism device-verified evidence — not just "diverges":
- **WRONG-VALUE** — values differ beyond per-dtype rtol/atol → kernel math bug. Pin the specific
  input→output pair on the device (e.g. `floor_divide(x, x) → 0` instead of 1).
- **SCALE:r** — `dev ≈ reference × r` (consistent factor, incl. sign flip) → a requantization-scale
  error in the op's own lowering.
- **ZEROED** — `dev` all-zero where the reference isn't → a dropped store in the kernel.
- **NONFINITE** — `dev` nan/inf where the reference is finite (and the leaf was finite) → overflow /
  precision loss in the op (common for fp16).
- **WRONG-SHAPE / WRONG-DTYPE** — output metadata wrong (after clearing the 3c artifact) → a
  meta/shape-inference bug in the op's lowering.
- **CRASH** — the op aborts alone → **record the crash reason**: the crashing form
  (dtype/shape/args) AND the runtime error string if any. A native abort usually has no catchable
  message (only `executor died (native abort)`) — say so explicitly, and note the trigger you can
  observe (e.g. non-finite input, negative/zero dim). The absence of a `Check failed` guard *is* the
  bug: the kernel should reject the form with a catchable error, not abort.
- **SKIP** — the op is rejected at runtime → **record the skip reason verbatim**: the runtime
  `Check failed (...)` / `not implemented` / `Missing operator` / `xnn_status_*` clause the backend
  raised, and enumerate the rejected forms (which dtypes/ranks/args SKIP vs run). The corpus job
  already *is* the one-op probe, so this is direct. Group by distinct reason signature — one op often
  SKIPs for several different reasons (e.g. an int64-index requirement vs a rank guard).

**Always write the reason strings down, not just the counts.** A SKIP/CRASH finding with only a count
is incomplete — the *reason* is the finding (which guard fired, or which guard is missing). Extract
them from the skip-log's `reason` column: it holds the full runtime error for every non-OK graph
(the kernel-level `[file.cpp:NN] Check failed (...)` clause is after the first `|`).

**Use the decomposition when `ops` disagrees with the source op count.** If a single source op
reports `ops=3` or `ops=5` delegated, it decomposed into several edge ops. Inspect the `.py` (and, if
needed, the lowered graph via `build_job`) to attribute the divergence to the specific edge op inside
the decomposition — still a finding for the source operator, but the mechanism names the internal op.

Mechanisms differ per backend (e.g. requant-scale errors on some, fp16 overflow on most). Derive it
from *this* backend's device values; don't assume it transfers.

## 5. Verify each root cause on the device
A mechanism hypothesis is not a finding until tested on the device. Test the proposed cause directly
(suspect "x/x rounds below 1" → run `div(x, x)` on device; suspect an op is unsupported → confirm its
one-op job SKIPs in every form). State only device-verified input→output pairs; mark anything
unpinned as not pinned.

**The predicate owns the oracle and env contract:**
- **Exact oracle** — use `net/compare.py` (`compare()`/`_cmp`): per-dtype rtol/atol, nan/±inf
  positions element-wise. If you index a single output, first drop ExecuTorch's PREPENDED
  input-mutation outputs — `if len(et) > len(eager): et = et[len(et)-len(eager):]` — before `et[T]`;
  skipping this silently flips verdicts on any `out=`/mutation graph.
- **Exact inputs** — reuse the graph's own `LEAVES`; data-dependent bugs (`floor_divide(x,x)→0`,
  fp16 overflow, OOB index) only survive on the same inputs. Don't substitute fresh random inputs.
- **Right backend + env** — rebuild repros with `build_job(src, <backend>)` (in `gen/export/job.py`,
  the same call the corpus was generated with — `pregen.py` is the working reference) under that
  backend's env, else it SKIPs everything and reads as "bug gone". Re-lower in the SAME environment
  the corpus was generated in:
  - **QNN/Qualcomm** (`run_pregen_qnn.sh`): `source android-dev/android-env.sh` first (exports
    `QNN_SDK_ROOT`, puts host `libc++`/libunwind + QNN libs on `LD_LIBRARY_PATH`); needs `flatc` on
    `PATH` (it's in `.venv/bin`). Runs in the 3.12 `.venv`.
  - **MediaTek** (`run_pregen_mtk.sh`): Python 3.10 `.venv-mtk` (mtk_converter is cp310-only); do NOT
    source android-env. See `neuropilot_sdk/README-venv-mtk.md`.
  - **Vulkan / XNNPACK / portable**: the 3.12 `.venv` with `flatc` on `PATH`; no extra SDK env.
  - Sanity-check the backend registered: `build_job` returns SKIP "unknown backend … (have: …)" when
    the env is wrong.
- **Stability** — gate every confirmed operator bug with the N≥5 re-runs from 3b.

## 6. Write each bug with a self-contained repro
One `bugs/<op>.md` + `bugs/repro_<op>.py` per confirmed operator bug. The repro builds the one-op
`.pte`, runs it on the device via the broker, and prints eager vs device (for SKIP/CRASH, prints the
device status). Label each with its **failure mode** (mismatch / crash / skip), its **mechanism**
(WRONG-VALUE / SCALE / ZEROED / NONFINITE / WRONG-SHAPE / WRONG-DTYPE — or, for SKIP/CRASH, the
**verbatim reason string**), and its **delegated-vs-portable** bucket. Record ops removed by the
step-3 filters (portable-fallback divergences, dtype-only artifacts, reference confounds, 0/N flaky)
under `ruled_out/` with the filter that ruled them out, so the report separates confirmed bugs from
suspects.

**SKIP and CRASH bugs must carry their reason, not just a count** (see step 4). Produce a
`skips.md` — a per-operator table of `count | operator | runtime reason` for every SKIP, plus a
CRASH table (`count | operator | crash trigger`, noting native aborts have no catchable message).
This *is* the SKIP/CRASH finding; a bare count is not.

## Coverage note — sizing the corpus so every op is caught
The seed schedule is exact round-robin over the ~228 usable ops, so each op is sampled
`k ≈ total / N` times. To catch each operator with enough samples to find form-specific bugs and
support the N≥5 determinism gate, size `--total ≥ 5·N` (≈ 1140), more for wider dtype/shape coverage.
Remember **scheduled ≠ produced**: constraint-heavy or backend-unsupported ops yield fewer (or zero)
jobs despite equal scheduling; those under-covered ops are the step-2 coverage gaps, not silent
successes. For a guaranteed floor per op regardless of attrition, generate with retry-until-k-successes
rather than a fixed attempt count.

## Outputs (this folder, per device)
- `README.md` — synthesis, with a **section per failure mode** (Mismatch / Crash / Skip), each entry
  labeled with its mechanism and delegated-vs-portable bucket, plus:
  - **Per-operator table** — the primary artifact: one row per (operator, form) with
    OK/MISMATCH/CRASH/SKIP counts over its `k` samples and the delegation breakdown.
  - **Coverage ledger** — ops scheduled vs produced vs verdicted; ops that never lowered
    (attrition / coverage gaps) enumerated.
- `bugs/` — each confirmed operator bug (md + runnable one-op repro), grouped by failure mode;
  `bugs/INDEX.md` lists them.
- `skips.md` — **required**: per-operator SKIP reason table (runtime `Check failed`/error verbatim)
  and CRASH reason table (crashing form + trigger). The reason is the finding for these modes.
- `ruled_out/` — suspects removed by the step-3 filters, with the reason.

## Rigor reminders (write these into every run)
- **On int8/low-precision, single-pass verdicts over-report several-fold — gate everything (3b).** A
  single genuine-looking isolation is not a confirmed bug; N≥5 collapses most near-tolerance noise.
  dtype divergence is usually an isolation artifact (3c). Filter reference-side int64/sentinel
  confounds before claiming a device bug (3d).
- **Legitimate SKIPs are real coverage gaps, not bugs** — e.g. an op with no portable kernel fails
  the load with "Missing operator". Enumerate these in the Skip section **with the verbatim reason
  string** (`skips.md`); don't chase them as failures, but don't drop the reason either — a count
  without the reason is not a finding. In this corpus they are trivial to identify — the job is
  already the one-op probe.
- **Record CRASH reasons too** — a native abort has no catchable message, so record the crashing
  operator + input trigger and say "native abort (no catchable error)". The missing guard is the bug.
- **On a real device, separate `executor unavailable` from real crashes, and re-run — don't count
  dropouts.** A flaky phone worker dropping under load logs its in-flight jobs as CRASH
  `executor unavailable` — these are NOT crashes. Split them out (grep the CRASH detail), and **re-run
  the whole CRASH set with lower in-flight concurrency** (drop `--window` from ~48 to ~6, raise
  `--timeout`) — a brief blip then loses far fewer in-flight jobs and the set converges in one pass.
  Merge the recovered verdicts (they hide real mismatches/skips). Counting dropouts as crashes once
  overstated a device's crash rate ~40×.
- **Long-lived helper processes** (broker, device/emulator clients) only survive when launched as a
  **bare single-line `exec <prog>`** via the harness background mechanism; a multi-line command
  (with `source …`/`set -x`, or a supervisor that `wait`s) gets torn down when the launching
  tool-call shell exits. One process per background task.
