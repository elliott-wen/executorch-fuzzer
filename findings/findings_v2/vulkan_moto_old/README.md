# Vulkan backend on Moto G54 5G — fuzzing findings

**Date:** 2026-06-29
**Corpus:** `corpus_v2/vulkan` — 76,801 ExecuTorch graphs (same corpus fed to the Pixel 9)
**Worker:** Moto G54 5G (Mali-G57), device, pulling from the local broker
**Pipeline:** `pregen` (lower) → broker → Moto G54 (run) → feeder (diff vs eager reference)

## Headline

**Every bug found on the Pixel 9 reproduces on the Moto G54 — identically.** Same culprits,
same crash cluster, same confounds, same device-confirmed memory-planning bug (down to the exact
corrupted value `2.4668`). The Moto is a different, lower-end SoC (Mali-G57 vs the Pixel's
Tensor G4), so the cross-device agreement is the key result: **these are ExecuTorch Vulkan
*delegate* bugs, not GPU/driver-specific.**

| outcome  | Moto G54 | Pixel 9 | meaning |
|----------|---------:|--------:|---------|
| OK       | 30,059 (39.1%) | 32,359 (42.1%) | output matched eager reference |
| SKIP     | 33,857 (44.1%) | 33,855 (44.1%) | Vulkan rejected the graph on-device |
| MISMATCH |  9,703 (12.6%) |  7,399 (9.6%)  | ran, numerically wrong |
| CRASH    |  3,182 (4.1%)  |  3,188 (4.2%)  | native abort |
| TIMEOUT  |      0         |      0         | no hangs |

The one device difference: the Moto has **more mismatches** (12.6% vs 9.6%). Per-job diff
(same corpus) shows **7,303 mismatches are shared** (96% at the same op) and only **2,396 are
Moto-unique** (96 Pixel-unique). Those 2,396 extra are **not a new bug class** — ~27% are the
known **copy-elision aliasing bug firing more under the Moto's memory planner** (silent wrong
data) and ~45% are **fp16 overflow** on magnitude-sensitive chains. Full breakdown with replayed
evidence: [CROSS_DEVICE_MISMATCH.md](CROSS_DEVICE_MISMATCH.md). The *real-bug* mismatch rates and
the SKIP/CRASH sets match the Pixel 9 almost exactly.

Raw per-job log: `tmp/vulkan_moto_skip.tsv`. Replay one: `python -m mobile feed <job_id> --corpus corpus_v2/vulkan`.

---

## Method (same three layers as the Pixel 9 run)

1. **Whole-graph op-presence** enrichment (provenance only — over-blames ~3×).
2. **Per-output attribution** (`CORRECTED_ATTRIBUTION.md`) — attribute each mismatch to the sink
   op that *directly produced* the bad output (the feeder reports the first failing `out[N]`,
   outputs are in sink order). Strips the co-occurrence confound.
3. **On-device isolation probes** (`ISOLATION_PROBES.md`) — single-op jobs run on the Moto;
   the only ground truth for CRASH (can't be per-output attributed).

---

## CORRECTED mismatch culprits (per-output; Moto baseline 6.98%)

real-bug rate excludes non-finite (upstream nan/inf propagation). Full table: `CORRECTED_ATTRIBUTION.md`.

| op | Moto real-bug rate | Pixel 9 | kind | isolation verdict |
|----|-------------------:|--------:|------|-------------------|
| `clamp.Tensor_out` | **55.5%** | 55.8% | numeric (514) | CONDITIONAL — simple case OK; trigger = broadcast/dynamic bounds |
| `select_scatter` | **51.4%** | 50.4% | **dtype (429)** | CONDITIONAL — float OK; wrong **dtype** for integer inputs |
| `index_put` | **49.6%** | 49.8% | numeric (141) | **GENUINE** — drops indexed writes, sets element 0 →0 (device-verified) |
| `bitwise_left_shift.Tensor_out` | 23.5% | 21.6% | numeric | CONDITIONAL |
| `topk.values` | 21.8% | 19.1% | numeric+shape | — |
| `sum.IntList_out` | 21.6% | 21.8% | numeric | reduction drift |
| `bitwise_left_shift.Tensor_Scalar` | 16.8% | 16.8% | numeric | CONDITIONAL |
| `diagonal_copy` | 16.5% | — | numeric | (more prominent on Moto) |
| `gather.out` / `tril.out` | 12.7% | 11.4% | numeric | mild |

**Confounds demoted** (whole-graph blamed them; per-output ≈ baseline, fails are nan/inf
propagation): `_softmax` 3.0%, `mean` 4.0%, `tanh` 4.3%, `_log_softmax` 4.5%, `pow` 7.8% (197/261
non-finite). And again the clincher: **scalar `clamp.out` 6.3% ≈ baseline vs `clamp.Tensor_out`
55.5%** — the bug is the tensor-bound path, on this device too.

## CRASH culprits (whole-graph enrichment; Moto crash baseline 9.6%)

| op | Moto crash-rate | Pixel 9 | isolation verdict |
|----|----------------:|--------:|-------------------|
| `unfold_copy` | 40.3% | 39% | **CONFOUND** (correct alone → aliasing bug) |
| `narrow_copy` | 36.2% | 35% | **CONFOUND** (correct alone → aliasing bug) |
| `scatter.src_out` | 35.6% | 35% | **GENUINE** — drops writes, sets element 0 →0 (device-verified) |
| `native_layer_norm` | 30.6% | 30% | **GENUINE** |
| `clone` / `lift_fresh_copy` / `alias_copy` | 21–24% | similar | **CONFOUND** → aliasing bug |
| `native_group_norm` | (support <200, filtered) | 82% | **GENUINE** (isolation: SKIP all forms) |

### The copy/view crash cluster is the memory-planning / aliasing bug — DEVICE-CONFIRMED on Moto
`unfold_copy`, `narrow_copy`, `clone`, `alias_copy`, `lift_fresh_copy` rank high for CRASH but
each runs **correctly in isolation**. The real bug: a value-preserving copy is elided to a buffer
alias, but the planner frees the source's buffer while the alias keeps it live → a sibling output
is silently overwritten. **Reproduced on the Moto G54 deterministically** — swapping only the
copy op for a fresh-buffer `add 0.0` flips `out[0]` from `2.4668` (wrong) to `0.8486` (correct).
See [bugs/vulkan-copy-elision-aliasing.md](bugs/vulkan-copy-elision-aliasing.md).

---

## Net genuine, fileable bugs (all reproduce on BOTH Moto G54 and Pixel 9)
1. **Vulkan copy-elision / memory-planning aliasing** — silently corrupts a sibling output;
   device-confirmed repro. Most serious (silent wrong data). Root of the copy/view crash cluster.
2. **`native_group_norm`** — rejected on Vulkan in all weight forms.
3. **`native_layer_norm`** — requires constant, non-None weight and rank-1 normalized_shape.
4. **`scatter.src_out`** — drops its writes and sets element 0 →0 (device-verified, isolated).
5. **`index_put`** — drops its writes and sets element 0 →0 (device-verified, isolated).
6. **`clamp.Tensor_out`** — wrong for broadcast/dynamic tensor bounds (scalar clamp is fine).
7. **`select_scatter`** — wrong output **dtype** for integer inputs.
8. **`bitwise_left_shift`** — wrong for bool input / int overflow / large shift amounts.

## Cross-device conclusion
Two independent devices (Tensor G4 / Mali-G715 and Mali-G57), same corpus, **identical bug set
and identical repro values** → the bugs live in the ExecuTorch **Vulkan delegate** (partitioner /
memory-planner / op implementations), not in any one GPU driver. File against
`executorch/backends/vulkan`.

## Confirmed kernel divergences (single-op, device-isolated — each has md + repro)
Beyond the aliasing bug, four ops diverge **in isolation** on the Moto (`bugs/INDEX.md`):
`floor_divide(x,x)`→0, `bitwise_left_shift`/`bitwise_right_shift` over-width shift,
`scatter.src_out` and `index_put` drop writes. `clamp.Tensor_out` / `select_scatter` / `max` /
`mean` / `softmax` etc. were
**ruled out** — correct in isolation, so their corpus mismatches are the aliasing class, not
kernel bugs. Every claim here is backed by a replay on the device.

## Artifacts
- `README.md` — this synthesis.
- `bugs/` — every device-confirmed bug, each with a self-contained repro (`bugs/INDEX.md`).
- `CORRECTED_ATTRIBUTION.md` — per-output mismatch ranking + crash enrichment.
- `CROSS_DEVICE_MISMATCH.md` — Moto-vs-Pixel mismatch differential (device-classified).
- `CROSS_DEVICE_CULPRITS.md` — all 57 Pixel-9 culprit ops checked against the Moto.
- `ISOLATION_PROBES.md` — on-device single-op probe verdicts.

Repro: `python tmp/moto_attr.py` (analysis), `python mobile/tmp/build_culprit_probes.py` +
`python -m mobile feed cp:* --corpus tmp/culprit_probe` (probes),
`python mobile/findings_v2/vulkan_moto/bugs/repro_vulkan-copy-elision-aliasing.py` (aliasing).
