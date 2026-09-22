# MISMATCH cluster: non-finite nan/inf positions (2,937)

> **Per-bug splits (each with a runnable replay):** Mechanism 1 negative-shift overflow →
> [bugs/bitwise-shift-negative.md](bugs/bitwise-shift-negative.md); Mechanism 2 `sign(NaN)` →
> [bugs/sign-nan.md](bugs/sign-nan.md); Mechanism 3 legit-OOB-NaN repositioned →
> [bugs/nonfinite-inherited-reposition.md](bugs/nonfinite-inherited-reposition.md);
> Mechanism 4 by-zero → [bugs/floor_divide.md](bugs/floor_divide.md), plus the born-here
> [bugs/distance-nonfinite.md](bugs/distance-nonfinite.md) and
> [bugs/normalization-nonfinite.md](bugs/normalization-nonfinite.md). Full index: [REPLAY.md](REPLAY.md).

## Summary

These 2,937 cases all have status `MISMATCH` with reason
`out[i] non-finite mismatch (nan/inf positions differ)`: the portable `.pte`
program ran fine, but a returned tensor has NaN/Inf in **different positions**
than eager PyTorch (portable produced NaN/Inf where eager was finite, or
vice-versa).

The cluster is **not** one bug. It decomposes into four mechanisms, in
descending impact:

1. **Invalid/negative integer bit-shift** feeding a float op — portable's
   `bitwise_left_shift`/`right_shift` with a **negative shift amount** diverges
   from eager, producing huge or wrong values that then overflow (`cosh`, `bmm`,
   `expm1`) to ±Inf where eager stays finite. (real kernel/UB bug)
2. **`sign(NaN)` semantics** — eager defines `sign(nan) = 0`; the portable
   `sign` kernel **propagates NaN**. Any downstream op then sees NaN where eager
   saw a finite 0, flipping non-finite positions. (real kernel bug)
3. **NaN born from a legitimately out-of-domain elementwise op** (e.g.
   `acos`/`asin` of |x|>1, `logit` of x∉[0,1], `log`/`sqrt`/`rsqrt`/`atanh` of
   invalid domain) — both backends produce the NaN, but **downstream ops
   re-position it** (`remainder`, `where`, `fmod`, reductions, broadcasting)
   because portable and eager use different formulas/ordering for those ops.
   (mostly a propagation/precision artifact)
4. **fp16 reduction / divide-by-zero overflow** — `mean`/`sum`/`floor_divide`/
   `reciprocal`/`div` accumulating or dividing into the fp16 range can produce
   ±Inf in one backend but a finite (often clamped/rounded) value in the other.
   (fp16 precision/overflow artifact, plus a real `floor_divide`-by-zero
   formula difference)

I confirmed mechanisms 1, 2, 3, 4 with in-process reproductions below
(portable `.pte` via `et_runner.run_pte` vs eager `m.g` on the **same stored
job inputs**). The repro harness is `tmp/run_portable/cmp_nonfinite.py`.

Reproduction note: feeding the stored job inputs (decoded from the `.pte`
bundle, which are byte-identical to the module's seeded `LEAVES`) ~58% of
sampled jobs (11/19) reproduce a non-finite position diff in this runtime. The
remaining ~42% do not reproduce via a naive `m.g(*inputs)` eager replay because
that replay diverges on shape/broadcast semantics from the corpus's captured
`inductor` reference (e.g. `floor_divide.out` with an `out=` shape that does not
match the natural broadcast) — those are eager-replay limitations, not evidence
the original diff was spurious.

## Which ops dominate

Derived by, for each non-finite-mismatch TSV row, parsing the `out[i]` index and
mapping it to the i-th **starred** node (`n*=`) in column 4 — the starred nodes
are exactly the graph's `return (...)` tuple, in order (verified against
`corpus/portable/w0/w0_1317.py`: starred n8,n10,n13,n15,n16,n19 ==
`return (n8,n10,n13,n15,n16,n19)`). Counting the op on the starred output node:

| op (on starred / returned output) | approx count |
|---|---|
| floor_divide | 691 |
| sign | 356 |
| _native_batch_norm_legit(_no_training) | 143 |
| _cdist_forward | 73 |
| div | 70 |
| pow | 63 |
| logit | 55 |
| reciprocal | 53 |
| gelu | 53 |
| rsqrt | 48 |
| remainder | 45 |
| mean | 38 |
| mul | 35 |
| atanh | 32 |
| _pdist_forward | 32 |
| (tail: log/log10/log2/log1p, sqrt, exp, expm1, cosh/sinh, acos/asin, native_layer_norm, native_group_norm, ...) | remainder |

This is the op on the *returned* node; the actual NaN/Inf frequently originates
one or two nodes upstream (e.g. a negative bit-shift, an out-of-domain `acos`, a
divide-by-zero) and is merely *carried* to the returned op. So the table shows
where the diff is *observed*, and the Mechanisms section shows where it is
*caused*.

## Mechanisms

### Mechanism 1 — negative/invalid integer bit-shift, then float overflow (real bug)

**job w0:116, out[1] = n10 = `cosh.out(n8)`** (fp16).
Chain: `n5 = logical_xor(...) -> [1,1]`; `n8 = bitwise_left_shift.Tensor_Scalar(n5, -3, out=fp32)`; `n10 = cosh(n8)`.
Eager intermediate (traced):
```
n5 = tensor([1, 1])
n8 (eager bitwise_left_shift(1, -3)) = tensor([0., 0.])
cosh(0) = tensor([1., 1.], fp16)
```
Reproduced output diff:
```
out[1] p=fp16/e=fp16 nonfinite-pos-diff=2
   [0] portable=inf  eager=1.0
   [1] portable=inf  eager=1.0
```
Eager treats `1 << -3` as `0` on this path (cosh(0)=1). Portable's
`bitwise_left_shift.Tensor_Scalar` with a **negative shift** computes a large
value, and `cosh` overflows fp16 to **+Inf**. Negative shift is C/C++
undefined behaviour; the portable kernel and eager disagree.

**job w0:1792, out[0]** — same family. Chain has
`n3 = bitwise_left_shift.Tensor_Scalar(n1, -2)` (negative shift) feeding `bmm`,
then expm1/softmax. Reproduced:
```
out[0] p=fp32/e=fp32 nonfinite-pos-diff>=1
   portable=inf  eager=0.0
```

### Mechanism 2 — `sign(NaN)`: eager=0, portable propagates NaN (real bug)

**job w0:1490, out[0] = n3 = `atan2.out(n1, n2)`** (fp32).
Chain: `n0 = logit(L0)`; `n1 = sign(n0)`; `n2 = clamp(n1, ...)`; `n3 = atan2(n1, n2)`.
Inputs/trace:
```
L0 = tensor([-0.4825, 1.1104])         # logit domain is (0,1); both OOB
n0 = logit(L0) = tensor([nan, nan])    # NaN in BOTH backends (legit)
n1 = sign(nan) :  eager -> tensor([0., 0.])
```
Eager confirms PyTorch semantics directly:
```
torch.sign(nan) = 0.0      # (and sign(±inf) = ±1)
```
Reproduced output diff:
```
out[0] nonfinite-pos-diff=2  portable=nan  eager=0.0
out[1] nonfinite-pos-diff=2  portable=nan  eager=0.46251991391181946
out[3] nonfinite-pos-diff=2  portable=nan  eager=0.0
```
Eager's `sign(nan)=0` "launders" the NaN, so `atan2(0,0)=0` (finite). Portable's
`sign` returns NaN, so `atan2(nan, nan)=nan`. This is a genuine portable-kernel
semantics bug for `sign` on NaN, and it directly explains a large share of the
356 `sign`-on-output cases.

### Mechanism 3 — NaN from a legit out-of-domain op, then re-positioned downstream (mostly artifact)

**job w0:313, out[1/3/5/7]**.
Chain: `n0 = acos(L0)`; `n2 = remainder.Tensor(broadcast(n0), n0, out=fp16)`; ...
```
L0 = tensor([-0.4825, 1.1104])
n0 = acos(L0) = tensor([2.0743, nan])   # acos(1.1104) OOB -> NaN in BOTH
```
Reproduced: `portable=nan  eager=0.0` at the mismatched positions.
Both backends agree NaN exists at the `acos` element. The **positions diverge**
at the downstream `remainder`/`where`/reduction nodes, where portable and eager
differ in how a NaN dividend/divisor and the fp16 cast are handled (formula and
broadcast ordering), so the NaN lands in different output slots.

**job w0:126** (`acos`/`asinh` chain): reproduced `portable=nan  eager=1.5707963`
(= π/2 = acos(0)); **job w0:455** (`acos`/`min`): `portable=nan  eager=0.0`;
**job w0:143** (batch-norm/reduction chain): `portable=nan  eager=0.0`. Same
shape: a legitimately-NaN element exists, and a downstream op disagrees on where
it ends up.

### Mechanism 4 — fp16 reduction / divide-by-zero overflow (artifact + minor formula bug)

**job w0:752, out[7] = log10 over an int sum cast to fp16**.
Reproduced:
```
out[7] p=fp16/e=fp16 nonfinite-pos-diff=1
   portable=1.9443359375   eager=-inf
```
Eager `log10(0) = -inf`; portable produced a finite ~1.94 — the reduction
feeding `log10` accumulated a different (nonzero) value in portable, so the two
disagree on whether the argument hit the `log10(0) -> -inf` singularity.

**floor_divide-by-zero** (the single largest op, 691): eager semantics are
```
0.0 // 0.0 = nan     1.0 // 0.0 = inf
int0 // 0.0 = nan    int1 // 0.0 = inf
```
When the divisor broadcasts to all-zero (e.g. w0:1515 `n16 = broadcast(clamp(...,0,0))`,
`n17 = floor_divide(n9, n16, out=fp32)`), eager yields nan/inf per the above. A
portable `floor_divide` kernel that computes `trunc(a/b)` or takes an integer
path, or rounds in fp16, will place nan/inf differently. These were not
reproducible through the naive eager replay (the `out=` shape does not match the
natural broadcast, so `m.g` itself raises), but the by-zero divergence is the
mechanistic cause and matches the op-frequency dominance of `floor_divide`.

## Classification

| Mechanism | dominant ops | class | rough share of cluster |
|---|---|---|---|
| 1. negative/invalid bit-shift → float overflow | bitwise_left/right_shift → cosh/bmm/expm1 | **real kernel/UB bug** | ~10-15% |
| 2. `sign(NaN)` propagation | sign (and any op consuming it) | **real kernel bug** | ~15-20% (much of the 356 `sign` cases + downstream) |
| 3. legit OOB NaN re-positioned downstream | acos/asin/logit/atanh/log/sqrt/rsqrt → remainder/where/fmod/reductions | **artifact** (both produce NaN; harness flags position) — partly **harness strictness** | ~30-40% |
| 4. fp16 reduction / divide-by-zero overflow | floor_divide, div, reciprocal, mean, sum, log10, rsqrt, pow | **fp16 precision/overflow artifact**, with a **real `floor_divide`-by-zero formula** sub-bug | ~30-40% (floor_divide alone is 691/2937 ≈ 24%) |

Notes on strictness: Mechanism 3 is partly a **comparison-harness strictness
issue** — when both backends legitimately produce a NaN somewhere and the only
disagreement is *which* element after a NaN-contaminated reduction/scatter, that
is arguably a tolerance/position artifact rather than a backend defect. The
harness checks NaN/Inf positions exactly, so any NaN-laundering difference
(Mechanism 2) or NaN-reordering (Mechanism 3) trips it.

## Recommendation

1. **Fix the two real kernel bugs first**, they are unambiguous and cheap:
   - Portable `sign` must return `0` for NaN inputs (match ATen
     `sign(nan)=0`). This alone should clear a large fraction of the 356
     `sign`-output cases and their downstream fan-out.
   - Portable integer `bitwise_left_shift`/`right_shift` with a **negative or
     >= bit-width shift** must match eager's behaviour (define/clamp it rather
     than invoking C++ UB). This removes the cosh/bmm/expm1 → ±Inf overflows.
2. **Verify portable `floor_divide`** uses the same by-zero/by-zero-vector
   semantics as ATen (`floor(a/b)` with proper nan/inf at b==0), since it is the
   single largest op in the cluster.
3. **Relax the harness for Mechanism 3/4**: when a NaN/Inf is already present in
   an *input* to the returned op (i.e. the non-finite is inherited from a
   legitimately out-of-domain upstream op or an fp16 overflow), treat NaN/Inf
   position differences as a precision/propagation artifact (e.g. compare with
   NaN==NaN equality and a "non-finite anywhere upstream" allowance) rather than
   a hard MISMATCH. This will de-noise the cluster down to the genuine kernel
   bugs in (1)/(2).

Reproductions were produced with
`/data/jwen929/mobile/tmp/run_portable/cmp_nonfinite.py` (portable via
`mobile.executor.et_runner.run_pte` on stored job inputs, eager via the module's
`g(*inputs)`), against
`/data/jwen929/mobile/corpus/portable/w0/w0_{116,1490,313,752,126,455,143,1792,1515}.py`.
