# MISMATCH family — QNN runs but diverges from eager (14,672 jobs, 38%)

Localized first-divergent op for a 397-job sample (354 pinned). Each bug below was deep-dived
(`deep_dive.py`) to the exact op call, triggering input values, and eager-vs-QNN outputs.
Kinds among pinned: **delta 61% · nonfinite 34% · dtype 2.5% · shape 2.5%.**

---

## BUG 1 — `sub`/`rsub` silently drop the `alpha` multiplier  ⚠️ HIGH

QNN computes `a − b` instead of `a − alpha·b`; the `alpha` scalar is ignored.

- **`sub`** `w16:1991`: `sub(n3=2.2676, n2=1.5574, alpha=-6)` → eager `2.2676 − (−6)(1.5574) = 11.61`, **QNN `0.71` (= 2.2676 − 1.5574)**. max|Δ| 10.9.
- **`rsub`** `w11:469`: `rsub.Scalar(n2=0.1124, other=5, alpha=4.0)` → eager `5 − 4·0.1124 = 4.55`, **QNN `4.89` (= 5 − 0.1124)**. max|Δ| 1.75.

Mechanism: the HTP add/sub lowering binds only the two tensors and the scalar `other`, not
`alpha`. **Silent wrong arithmetic whenever `alpha ≠ 1`** — the most dangerous class (no nan,
no crash, just wrong numbers). ~46 jobs in the sample (sub 14 + rsub 16 + add-family).

---

## BUG 2 — `_to_copy` float→int cast ROUNDS; eager TRUNCATES  ⚠️ HIGH

`w0:1032`: `_to_copy(n4: float32 → int64)`, input `[1.9166, 0.3436, −0.5795, −0.9276]`
- eager (truncate toward 0): `[1, 0, 0, 0]`
- **QNN (round to nearest): `[2, 0, −1, −1]`**

Mechanism: HTP cast uses round-to-nearest; PyTorch `to(int)` truncates toward zero. Off-by-one
on every non-integer magnitude ≥0.5. 37 jobs — the single largest mismatch op.

---

## BUG 3 — Non-finite results collapse to finite (34% of mismatches)

Eager produces `nan`/`inf` on out-of-domain / div-by-zero; the HTP returns a finite value.

| Op | Job | Input | eager | QNN |
|---|---|---|---|---|
| `log` | `w1:2142` | `log(0)` (bool False) | `−inf` | **`0.0`** |
| `div` | `w15:1282` | `x / 0` (bool False divisor) | `±inf` (24 pos) | **large finite** (584, −45696, …) |
| `rsqrt` | `w11:1329` | `rsqrt(−0.84)` (neg) | `nan` (36) | **partially finite** (29 nan; e.g. −2.5e−5) |
| `logit` | `w0:735` | `logit(inf)` | `nan` | **`−8.6875`** |
| `pow` | `w14:1877` | `pow(−5, nan)` | `nan` | **`−98304.0`** |

Mechanism: the HTP kernels lack the IEEE special-case paths (domain checks, div-by-zero → inf,
nan propagation). `rsqrt` is *inconsistent* (some lanes nan, some finite) — a vectorization/
approximation artifact. Flagged by the differ as `non-finite mismatch (nan/inf positions differ)`.

---

## BUG 4 — `bitwise_left_shift` with a negative shift → garbage  ⚠️ HIGH

`w19:2365`: `bitwise_left_shift.Tensor_Scalar(L0: bool, −4)`
- eager: `[0, 0, 0]` (negative shift defined as 0)
- **QNN: `[0, 1.153e18, 1.153e18]`** (≈ 2^60 — undefined-behavior shift)

Mechanism: HTP shifts by the raw (negative→huge unsigned) count instead of clamping. 14 jobs.
(Output dtype here is `float32` for a bitwise op — a generator artifact, but the divergence is
real and integer-domain.)

---

## BUG 5 — `elu` mishandles non-default `alpha`/`scale`/`input_scale`

`w21:932`: `elu(n6=0.5973, alpha=−5, scale=−2, input_scale=−6)` (positive input)
- eager `scale·x = −2·0.5973 = −1.195`
- **QNN `−3.488`**. max|Δ| 2.29.

Mechanism: for `x>0`, `elu = scale·x`; QNN appears to apply the negative branch / mis-fold the
three scale params. 12 jobs. Only triggers with non-default (here negative) scale params.

---

## BUG 6 — `select_scatter` output dtype differs (5, dtype)

QNN returns a different scalar-type than eager for the scattered output (dtype-kind divergence).
Lower frequency; see `../localize/mismatch_localized.tsv` (`$3==dtype`).

---

## Severity ranking
1. **Silent-wrong-math** (BUG 1 `alpha`, BUG 2 cast-rounding, BUG 4 neg-shift) — no signal, wrong values.
2. **Non-finite collapse** (BUG 3) — detectable (nan/inf), but masks real eager behavior.
3. **Param-sensitive** (BUG 5 elu) — only off the default config.

## Reproduce any
```bash
source android-dev/android-env.sh
export PYTHONPATH=/data/jwen929 LD_LIBRARY_PATH="$PWD/pytorch_ref/executorch/build-x86/lib:$LD_LIBRARY_PATH"
.venv/bin/python findings_v1/qnn_local/deep_dive.py corpus_v1/qnn/w16/w16_1991.py   # the alpha bug
```
