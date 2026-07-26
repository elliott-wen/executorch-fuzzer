# Ruled-out suspects — divergences that are NOT Core ML operator bugs

Every row here isolated to one operator but was removed by a step-3 filter of the single-op analysis
workflow. Keeping them separate is the point: it stops the confirmed list (bugs/) from being inflated
by confounds. Each is device-checked.

## A. Portable-fallback mismatches (`delegated.ops = 0`) — the delegate never ran the op
These graphs lowered READY but Core ML did **not** absorb the op; the device executed the **portable
CPU kernel**. A divergence is a portable-kernel/reference issue, not a Core ML bug. (Filter 3a.)

| operator | portable mismatches |
|----------|--------------------:|
| `bitwise_left_shift.Tensor_Scalar` | 199 |
| `bitwise_left_shift.Tensor_out` (portable share) | 123 |
| `grid_sampler_2d` | 81 |
| `bitwise_left_shift.Tensor_Scalar_out` | 72 |
| `prod.out` | 61 |
| `prod` | 50 |
| `bitwise_right_shift.Tensor_Scalar` | 20 |
| `fmod.Scalar` | 18 |
| `_pdist_forward` (portable share) | 2 |
| `bitwise_right_shift.Tensor_Scalar_out` | 1 |

> Note: `bitwise_left_shift.Tensor_out` has **both** a portable share (ruled out) and a genuine
> delegated share (20, confirmed — see [../bugs/bitwise-left-shift.md](../bugs/bitwise-left-shift.md)).

## B. Empty-tensor reduction — reference sentinel (filter 3d)
Reductions over a length-0 tensor: the eager CPU reference returns the dtype identity sentinel, the
device returns 0. The reference is degenerate, not the device.

| operator | job | eager | device |
|----------|-----|-------|--------|
| `min.unary_out` | `w102:432` | `9.223e18` (INT64_MAX) | `0` |
| `min.unary_out` | `w104:460` | `2.147e9` (INT32 sentinel) | `0` |
| `max.unary_out` (mismatch form) | `w101:656` | `-2.147e9` (INT32_MIN) | `0` |
| `mean.out` (mismatch form) | `w75:512` | `nan` (empty→nan) | `0` |

> `max.unary_out` also has a genuine **hang** form (see [../bugs/hang-timeout.md](../bugs/hang-timeout.md));
> only its empty-reduction *mismatch* is ruled out here.

## C. Non-finite INPUT propagation (filter 3e exception)
The leaf input was **already non-finite** (the corpus injects `inf`/`nan`), so a nan-vs-inf divergence
is just eager and Core ML propagating the non-finite input differently — not a kernel manufacturing a
wrong finite value. Leaf verified non-finite.

| operator | job | leaf | eager → device |
|----------|-----|------|----------------|
| `gelu.out` | `w100:154` | `L0=[inf,inf]` | `nan → inf` |
| `floor_divide` / `floor_divide.out` | `w100:12` / `w1:758` | `L0=inf` | `nan → inf` |
| `prod.int_out` (mismatch form) | `w11:71` | `L0=inf` (scalar) | `0 → 255` |
| `pow.Tensor_Tensor_out` (non-finite form) | `w30:169` | degenerate `0^neg` | `0 → inf` |
| `_native_batch_norm_legit_no_training` | `w6:278` | near-zero variance (0/0) | `nan → inf` |

> `prod.int_out` has a genuine **crash** finding; only its non-finite *mismatch* form is ruled out.

## D. Collateral crashes (filter 3b) — logged CRASH in bulk, re-ran clean
When one graph natively aborts a worker, its in-flight neighbours are also logged `native abort`.
Re-running each at `--window 1` (+ N=5 for borderline) flipped these back to OK/SKIP. **~48% of the
268 bulk CRASH rows were collateral.** Fully-collateral operators:

`mul.out` (0/5), `abs`, `bmm.out`, `native_layer_norm`, `minimum.out`, `sign.out`, `eq.Tensor_out`,
`logical_not.out`, `logit.out`, `squeeze_copy.dims`, `max.dim_max`, `min.dim_min`, `_log_softmax.out`,
`bitwise_xor.Scalar_out`, `split_copy.Tensor`, `unsqueeze_copy`, `erf.out`, `copy`, `mean`,
`ne.Scalar_out`, `bitwise_not.out`, `where.self`, `where.self_out`, `clone`, `pow.Scalar_out`,
`min.unary_out`, `logical_not`, `masked_fill.Scalar`, `logical_xor.out`, `squeeze_copy.dim`,
`mul.Scalar`, `reflection_pad2d`, `sigmoid.out`, `addmm.out`, `tan.out`, `gt.Tensor_out`, `acos.out`,
`unbind_copy.int`, `relu`, `select_copy.int`, `expm1.out`.

(`avg_pool2d.out`'s bulk "crashes" were mostly **SKIP** — static-tensor resize — with one genuine
crash; its real findings are the mismatch + SKIP, see [../bugs/avg_pool2d.md](../bugs/avg_pool2d.md).)
