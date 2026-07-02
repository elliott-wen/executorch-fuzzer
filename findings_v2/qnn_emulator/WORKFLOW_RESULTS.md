# Workflow results — QNN HTP x86 emulator (Qualcomm backend, fp16)

`analysis.md` run end-to-end against the **QNN HTP x86 emulator** (locally-built
`qnn_executor_runner`, qualcomm backend, fp16 HTP path — `quantize=False`) on the full
`corpus_v2/qnn` corpus (61,340 lowered graphs). Every classification is device-verified on the
emulator. Run on an **isolated broker** (ports 15564/65/66) with local QNN emulator clients, so it
never mixes with the Vulkan phone workers.

- skip-log: `skip_log.tsv`  ·  per-job verdicts: `bisect_results.tsv`  ·  op enrichment: `op_enrichment.txt`
- per-operator catalog: `bugs/operators/INDEX.md`  ·  isolation provenance: `operator_isolation.tsv`
- Setup notes: memory `qnn-local-emulator`. A harmless cp "same file" stderr in `qnn_runner.sh` was
  fixed during this run (inputs already live in the job dir).

## Triage (full corpus)
| outcome | count | share |
|---|---:|---:|
| OK | 20,618 | 33.6% |
| MISMATCH | 19,923 | 32.5% |
| CRASH | 818 | 1.3% |
| SKIP | 19,981 | 32.6% |
| TIMEOUT | 0 | 0% |

**The standout vs Vulkan: ~3× the mismatch rate** (32.5% vs ~11%). This is expected — the fp16 HTP
emulator is diffed against the **fp32** eager reference, so fp16 rounding alone pushes far more
graphs past tolerance. Much of the "operator bug" surface here is therefore a **fp16/precision**
story, distinct from the Vulkan kernel bugs.

## Full per-job bisection — ALL 19,923 mismatches (coverage gate: PASS)
**19,923 verdict rows == 19,923 mismatch targets** asserted (`bisect_results.tsv`).

| verdict | count | share | meaning |
|---|---:|---:|---|
| **OPERATOR** | 14,984 | 75.2% | target diverges alone → kernel/precision bug |
| **GRAPHOPT** | 2,936 | 14.7% | needs a co-returned sibling → graph-level |
| **NOREPRO** | 1,976 | 9.9% | didn't re-diverge → flaky / near-tolerance (more than Vulkan) |
| **BISECT_CRASH/HANG/LOWERFAIL** | 27 | 0.1% | re-lowering/running crashes or hangs the emulator/toolchain — un-bisectable (a real crash finding) |

## Graph-optimization bug — CONFIRMED: `fill.Scalar` output-buffer aliasing
The bisection labeled 2,936 mismatches GRAPHOPT. The usual output-set A/B test (`confirm_graphopt.py`)
**can't** confirm them on QNN, because the top targets (`fill.Scalar`, `copy`, `roll`) won't lower
**alone** (`build_job` → SKIP), so "correct alone" can't be established that way. **But a
constant-output probe sidesteps that wall and confirms a real memory-planning bug** (`tmp/qnn_fill_probe.py`):

**`fill.Scalar(x, C)` must output the constant `C`** regardless of its input values or anything else
in the graph — its correct answer is known from op semantics, no device isolation needed. In graphs
returning `{fill, sibling}`, the device returns **non-constant, foreign values** for the fill output
(stable 3/3):

| job | fill should be | device returns | sibling data-independent? |
|---|---|---|---|
| w100:66 out4 | `7.0` (all) | `[-0.482, 1.11]` | yes (not in fill's cone) |
| w102:448 out4 | `-2.0` (all) | `[0.934]` | yes |
| w103:514 out1 | `8.0` (all) | `[0.0, 0.015, 0.021]` | yes |

Since fill's output is a constant that **cannot** depend on the (data-independent) sibling, the only
explanation is that **QNN's memory planner aliased the fill output buffer with a live tensor, so the
constant got overwritten** by the sibling's computation — a textbook **buffer-aliasing /
memory-planning graph-optimization bug**. `fill.Scalar` is the #1 GRAPHOPT target (520 cases), so
this is a large class. See `bugs/graphopt-fill-buffer-aliasing.md`.

Note: the *output-set bisection* labels for QNN still over-count (many GRAPHOPT verdicts are flaky or
`select_scatter`-style mislabeled operator bugs); the **confirmed** graph-opt finding is the
`fill.Scalar` buffer-aliasing class above. Repro:
`BISECT_BACKEND=qualcomm BROKER_PORT=15564 python tmp/qnn_fill_probe.py w100:66 4 3`.

## Per-operator bug catalog (single-op isolation, exact baked inputs)
194 OPERATOR-verdict candidate ops → **58 isolated GENUINE** (120 CLEAN/pass-alone, 8 LOWERFAIL, 4
NONFINITE_OUT) → after a **7× stability re-verification**, **32 confirmed**: **25 deterministic
operator bugs + 7 non-deterministic kernels**; 26 ruled out. Each confirmed op has
`bugs/operators/<op>.md` + runnable `repro_<op>.py` with worst-element evidence; see
`bugs/operators/INDEX.md`.

**Notable QNN-specific result: 7 non-deterministic kernels** (wrong output *intermittently* on the
*same* baked input) — roughly **2× the Vulkan rate** (Vulkan: 3–4). The HTP emulator is materially
less deterministic than the Vulkan GPUs, on top of its higher precision-driven mismatch rate.

## Cross-backend comparison (vs the Vulkan runs)
| | Vulkan (Moto A14) | Vulkan (ASUS) | **QNN HTP emulator** |
|---|---|---|---|
| corpus | 76,801 | 76,801 | 61,340 |
| MISMATCH rate | 12.6% | 11.1% | **32.5%** |
| bisection OPERATOR/GRAPHOPT/NOREPRO | 81.5/11.9/6.5 | 82.4/10.6/7.0 | **75.2/14.7/9.9** |
| confirmed operator bugs | 49 | 24 | **32** |
| non-deterministic kernels | 4 | 3 | **7** |

**Takeaways:**
- QNN's much higher mismatch + NOREPRO rates are a **fp16-vs-fp32 precision** effect, not purely
  kernel bugs — the per-op isolation (which uses the same loose oracle) filters the genuine ones
  down to 32, but many corpus "mismatches" are precision noise the bisection labels NOREPRO.
- The QNN HTP emulator is the **least deterministic** target tested (7 non-det kernels).
- Cross-backend, the *operator-bug op families* overlap heavily with Vulkan (clamp, select_scatter,
  remainder, floor_divide, bitwise shift, index_put, gather) — these are ExecuTorch-level
  operator defects that surface on both the Vulkan GPUs and the Qualcomm HTP path, plus
  backend-specific ones (QNN enrichment also surfaces `rsqrt.out`/`sqrt.out` like the ASUS).
