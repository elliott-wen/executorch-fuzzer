# Ruled-out suspects — divergences that are NOT confirmed Ethos-U operator bugs

Each population below looks like a single-op divergence but is removed by a step-3 filter from
`analysis_single.md`. Kept here so the report separates confirmed bugs (`../bugs/`) from suspects.

## 1. Portable-fallback divergences — filter 3a (delegated-vs-portable)
**1,812 MISMATCH + 4,260 SKIP + 442 CRASH** occurred on jobs where the Ethos-U partitioner
declined the op (`delegated ops=0`) and the **portable CPU kernel** ran on the Cortex-M core. A
divergence there is a portable-kernel / reference issue, **not** an Ethos-U delegate bug, so it is
not filed against Ethos-U. Top portable MISMATCH ops (for reference — candidate *portable* bugs):

| op | portable MISMATCH |
|----|------:|
| `gelu.out` | 365 |
| `index_put` | 206 |
| `t_copy` | 140 |
| `bitwise_right_shift.Tensor_Scalar[_out]` | 268 |
| `transpose_copy.int` | 127 |
| `remainder.Scalar` | 83 |
| `grid_sampler_2d` | 81 |
| `prod` | 54 |
| `_native_batch_norm_legit.no_stats` | 45 |

136 of the 199 produced ops are **always portable** under Ethos-U (0 delegated jobs) — the entire
Ethos-U delegate surface is only 63 ops (see `../README.md` coverage ledger).

## 2. Non-finite-input divergences — filter 3e
**799** delegated MISMATCHes are driven by the corpus's domain-fuzz injecting `nan/inf/-inf` into a
float leaf. In a single-op quantized graph an inf/nan input is out of the quantization domain, and
the quantized reference (which clamps `inf→127`) diverges from the device's handling — this is a
non-finite-**input** propagation artifact, not a clean finite-domain kernel bug. Notably this
accounts for **every** copy/movement-op "mismatch": `select_copy.int` 124, `split_copy.Tensor` 58,
`relu` 42, `div.Scalar` 33, `permute_copy` 33, `mul.Scalar` 31, `abs` 28, `index_put` 205,
`unsqueeze_copy` 86, `mean` 18, `t_copy` 12, `pixel_unshuffle` 10, … (leaves verified non-finite).
`floor_divide` and `rsub.Scalar` each have *both* a non-finite population (ruled out) and a
FINITE-CLEAN population (confirmed — see `../bugs/`).

## 3. Negative-shift undefined behavior — filter 3d (reference quirk)
**794** delegated `bitwise_left_shift` / `bitwise_right_shift` MISMATCHes have a **negative** shift
operand (leaf values in `[-4,4]`). Shifting by a negative amount is undefined; the PyTorch CPU
reference returns a particular value (often 0) and the device returns a different one. A mismatch on
undefined behavior is not a correctness bug — ruled out. (The **non-negative** shift mismatches are
kept as a confirmed bug: `../bugs/bitwise-right-shift.md`.)

## 4. Intermittent / flaky
None. Every FINITE-CLEAN candidate that was determinism-gated reproduced **5/5** (`TIMEOUT=0` over
the whole run, so there were no dropouts to discard either).
