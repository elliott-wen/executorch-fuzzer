# CRASH family — genuine hard failures (629 jobs, 1.6%)

Post skip-aware-patch, CRASH = uncatchable hard failure (abort/segfault outside the runner's
clean error paths). All 629 localized in isolated subprocesses; representatives re-run through
the runner to capture the abort signature + op-chain. Raw: `../localize/crash_localized.tsv`.

## Process-death signature (full run, 629)
| Signal | Count | % |
|---|---:|---:|
| `rc=134` abort (SIGABRT) | 340 | 54% |
| `rc=139` **segfault** (SIGSEGV) | 286 | 45% |
| `rc=136` SIGFPE | 3 | 0.5% |

## Localizer outcome (629)
| Outcome | Count | Meaning |
|---|---:|---|
| DIED | 316 | crashes the localizer too → genuine hard crash |
| RESULT | 218 | all-node form diverges at an op first (same families as MISMATCH) |
| ERROR | 86 | localization raised (76 `execute() failed`, 7 context-binary, 2 vector-max, 1 bad_alloc) |
| OK | 9 | no divergence |

---

## BUG 10 — Heap corruption in the QNN backend  ⚠️ CRITICAL  (most of the rc=134)

Confirmed abort messages on representatives:

| Job | Abort | Op-chain (head) |
|---|---|---|
| `w0:1161` | **`free(): invalid pointer`** | `remainder → atan → expand_copy → round → trunc → logit` |
| `w1:121` | **`double free or corruption (out)`** | `bitwise_left_shift → bcast → addmm → mean → fmod` |
| `w1:91` | **`free(): invalid next size (normal)`** | `isnan → bitwise_right_shift → logical_xor → gt → fmod` |

Mechanism: the QNN backend / HTP runtime corrupts the heap — sometimes during **teardown after a
successful inference** (one case writes its etdump, then `free(): invalid pointer` on
`Destroy Qnn context`). Recurring ops in the corrupting chains: `fmod`, `bitwise_*`, `addmm`,
`mean`, `expand_copy`. These are **memory-safety bugs** — the highest-severity findings and the
right candidates for upstream reports (reproducible, deterministic per graph).

---

## BUG 11 — Segfaults  ⚠️ CRITICAL  (286, 45% of crashes)

`w14:30` → `rc=139` (SIGSEGV, no message). Op-chain: `max(L0) → logical_not → le → remainder →
ne → roll`. Pure null/oob dereference inside the HTP runtime; no diagnostic emitted. Op-chains
frequently feature reduce-to-scalar (`max`,`mean`) + `roll`/`expand` rank changes — overlapping
with the 6004 rank/batch family, but here it faults instead of returning an error.

---

## BUG 12 — Internal ET/QNN asserts  (≈43% of the DIED sample)

The DIED sample (n=60) shows `assert failed` aborts that are **not** from the (patched) runner —
they fire inside the QNN backend `.cpp`. These are graph constraints the partitioner should have
rejected at lowering time rather than asserting at runtime. Lower severity than 10/11 (they're
intentional aborts, not memory unsafety) but still surface as CRASH.

---

## BUG 13 — Localizable crashes mirror the MISMATCH bugs  (218 RESULT)

When run in all-node form, 218 "crashes" localize to a divergent op before dying: `_to_copy`
(21), `select_scatter` (6), `remainder` (6), `ne` (6), `logit` (6), `fill` (6), `slice_copy` (5),
`rsub` (5). Kinds: delta 173 / nonfinite 35 / shape 6 / dtype 4. Interpretation: these graphs
combine a numeric-divergence op (BUGs 1–5) with a downstream op whose handling of the bad
intermediate (e.g. a rounded/garbage value used as an index/shape) tips into a hard fault.

## Significance
- **BUG 10 (heap corruption) + BUG 11 (segfault) = ~95% of crashes** and are genuine
  memory-safety defects in the QNN runtime — file upstream with the repro graphs.
- BUG 12 asserts are a partitioner/runtime contract gap (admit-then-abort).

## Reproduce (isolated)
```bash
source android-dev/android-env.sh
export PYTHONPATH=/data/jwen929 LD_LIBRARY_PATH="$PWD/pytorch_ref/executorch/build-x86/lib:$LD_LIBRARY_PATH"
# capture the abort message + op-chain via the runner (preserves run.log):
.venv/bin/python findings_v1/qnn_local/localize_one.py corpus_v1/qnn/w0/w0_1161.py   # -> DIED
```
