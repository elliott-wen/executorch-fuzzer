# Investigation: normalization ops (batch_norm / group_norm / layer_norm / log_softmax) — non-finite divergence

## Verdict (up front)

**One REAL kernel divergence, born in the portable kernel: `_native_batch_norm_legit_no_training` / `_native_batch_norm_legit` (no_stats too).** When variance is **zero** (and `eps == 0`) the portable kernel computes `invstd = 1.0/std::sqrt(0) = +inf` and then `(x-mean)*invstd`, producing `±inf` (or `nan` only where `x==mean`). ATen/eager does **not** do this: its `InvStd` functor (`aten/src/ATen/native/Normalization.cpp:104-112`) explicitly guards `var==0 && eps==0` and forces `invstd = 0`, and the eager/inductor reference the fuzzer compares against produces a `nan`/`inf` **pattern that differs position-by-position from portable**. This is a finite-input → portable-differs-from-eager divergence and is a genuine kernel bug. It is triggered almost entirely by the fuzzer feeding **`eps=0.0` together with an arbitrary `running_var`** (zero or negative), which is the dominant driver of the 22.5% batch_norm propensity.

**The other three ops are NOT kernel bugs (artifact / correct propagation):**
- `native_group_norm`: portable matches eager (`nan==nan` on degenerate variance; finite values agree to fp tolerance). Its 100%-non-finite jobs are **inherited** — the non-finite enters the op's input from upstream.
- `native_layer_norm`: portable matches eager bit-for-bit on the degenerate (var=0 → `nan`) case. Representative job `w0:1569` feeds the norm an input derived from `log2` (already `-inf`/`nan`) → **inherited**.
- `_log_softmax`: portable matches eager exactly on `-inf` rows (`nan`) and `+inf` input (`nan`). The non-finite is **inherited/correctly propagated** upstream.

## Isolation evidence (the decisive test)

Minimal controlled export, identical inputs to both backends (`scratchpad/isolate.py`, `isolate2.py`):

### batch_norm `_native_batch_norm_legit_no_training(x, None, None, mean, var, momentum=0.0, eps)`

| case (x-mean, var, eps, dtype) | eager | portable | nan-pos differ | inf-pos differ |
|---|---|---|---|---|
| x-mean=2, var=0, eps=0, f32 | `nan` | `inf` | **yes** | **yes** |
| x-mean=-3, var=0, eps=0, f32 | `nan` | `-inf` | **yes** | **yes** |
| x-mean=0, var=0, eps=0, f32 | `nan` | `nan` | no | no |
| var=[1,0,-2], eps=0, f32 | `[2, nan, nan]` | `[2, inf, nan]` | **yes** | **yes** |
| var=[1,0,-2], eps=0, f16 | `[2, nan, nan]` | `[2, inf, nan]` | **yes** | **yes** |
| var=[1,0,-2], eps=1e-5, f32 | `[2.0, 632.4556, nan]` | `[2.0, 632.4555, nan]` | no | no |

Note: ATen's mix of `nan` vs `±inf` for `var=0` depends on the per-element `(x-mean)` magnitude/sign (verified directly: `bn([[1.16,0.51,0.67,0.10],[-0.13,-0.79,-0.91,-0.27]], mean=[0,1.5708,1.5708,1.5708], var=0, eps=0)` → eager `[nan,nan,nan,nan,nan,-inf,-inf,-inf]`). Portable instead yields all `±inf`/`nan` from a single `invstd=inf`. With **`eps>0`** (last row) negative/zero var no longer diverges — both go through `sqrt(var+eps)` identically; the divergence is specific to `eps==0` + degenerate var.

### group_norm / layer_norm / log_softmax (no divergence)

```
[gn const-channel eps=0 f32] nan-pos-differ=False  inf-pos-differ=False   (finite vals agree ~1e-6)
[gn constant input eps=0]    eager=[nan,nan,nan]  portbl=[nan,nan,nan]
[ln const eps=0 f32]         eager=[nan]*6        portbl=[nan]*6
[lsm -inf rows f32]          eager=[nan,nan,nan,-1.313,-inf,-0.313] == portbl  (exact)
[lsm +inf f32]               eager=[nan,nan,nan]  == portbl
```

### Lead job confirmation (`w0:1358`)

`_native_batch_norm_legit_no_training(L3, None, None, n5, n24, 0.0, 0.0)` with replayed inputs:
- `input L3` = finite `[1.16, 0.51, ... -0.27]`
- `mean n5` = `[0.0, 1.5708, 1.5708, 1.5708]` (finite)
- `var n24` = `[0.0, 0.0, 0.0, 0.0]`, `eps = 0.0`

So inputs are **finite**; the non-finite is **born in the op** from `var=0, eps=0`. This is exactly the isolated divergence above. (Direct ExecuTorch export of this exact subgraph hits an unrelated `0 in strides` error from the upstream broadcast, so the controlled isolation is the clean reproduction.)

## Kernel root cause (quoted source)

**Portable** `pytorch_ref/executorch/kernels/portable/cpu/op_native_batch_norm.cpp:119` (and identical at `:296` for the no_stats variant):
```cpp
CTYPE invstd = 1.0 / std::sqrt(var + eps);     // var=0,eps=0 -> 1/0 = +inf
...
*out_data = (*in_data - mean) * invstd * weight_val + bias_val;   // (x-mean)*inf -> ±inf / nan
```

**ATen** `pytorch_ref/aten/src/ATen/native/Normalization.cpp:104-112`:
```cpp
template<typename T>
struct InvStd {
  T operator()(T var, double epsilon) const {
    T invstd = 0;
    if (var != static_cast<T>(0) || epsilon != static_cast<T>(0)) {
      invstd = static_cast<T>(1) / std::sqrt(var + epsilon);
    }
    return invstd;     // var==0 && eps==0  ->  invstd = 0 (NOT inf)
  }
};
```
ATen's reference applies `((input - mean) * invstd) * weight + bias` (`Normalization.cpp:194`). The eager/inductor differential reference the harness uses produces `nan` (and a `nan`/`inf` mix), confirmed at the op level by both a direct `torch.ops.aten._native_batch_norm_legit_no_training` call and `torch.compile(backend='inductor')` (→ `nan`), versus portable `inf`. The single missing line — the `var != 0 || eps != 0` guard — is the entire root cause. For `var<0` both sides return `nan` (`sqrt(neg)`), so negative var alone does not diverge; the divergence is the **var==0 + eps==0** corner.

`group_norm` (`op_native_group_norm.cpp:87-88`) and `layer_norm` (via `layer_norm_scalar`) compute variance internally from the data and use the same `1.0/std::sqrt(variance+eps)`; with a constant input they get `variance=0` and `rstd=inf`, but `(x-mean)=0` there so they land on `nan` — matching ATen's `nan`, hence no observed divergence. They have no externally-supplied degenerate `var` to expose the `invstd=inf` path the way batch_norm's `running_var` does.

## Scope

- skip_reasons (`tmp/run_portable/skip_reasons_portable.tsv`, `MISMATCH` + `non-finite`): **batch_norm 452, layer_norm 446, log_softmax 155, group_norm 99** (= 963 norm non-finite mismatches recorded; the brief's 147/22.5% is the sampled per-op propensity slice).
- **Born-here (real kernel bug):** the batch_norm family, driven by `eps=0` + degenerate `running_var`. The propensity-sweep "143 non-finite / 147" for batch_norm is consistent with virtually all of them being this `eps=0`+bad-var pattern.
- **Inherited (artifact):** group_norm, layer_norm, log_softmax — non-finite arrives from upstream ops (`log2`, `asin`, etc.) or is correctly propagated; isolation shows position-exact agreement with eager.

## Fix / recommendation

1. **Kernel fix (recommended, low-risk):** port ATen's `InvStd` guard into the portable kernels. In `op_native_batch_norm.cpp` replace both `1.0 / std::sqrt(var + eps)` sites (lines 119 and 296) with a guarded form:
   ```cpp
   CTYPE invstd = (var != CTYPE(0) || eps != 0.0)
       ? CTYPE(1.0 / std::sqrt(var + eps)) : CTYPE(0);
   ```
   This makes portable bit-compatible with ATen for the `var==0, eps==0` corner and eliminates the inf-vs-nan divergence. (group_norm/layer_norm can adopt the same guard defensively for parity, though they don't currently diverge.)

2. **Harness:** `eps=0.0` combined with an invalid `running_var` (zero or negative) is a degenerate/undefined input — both backends produce non-finite, and ATen's own answer (`invstd=0` ⇒ output collapses to `bias`) is itself arbitrary. Recommend the fuzzer **either** (a) constrain `_native_batch_norm_legit*` generation to `eps > 0` and `running_var >= 0` (matches real model usage), **or** (b) classify `eps==0 ∧ min(running_var) <= 0` as a SKIP (degenerate-arg), the same way other div-by-zero corners are handled. Option (a) is preferable: it keeps the op in the fuzz surface while removing the undefined region, and any remaining batch_norm non-finite mismatch would then be a genuine signal.

3. With the kernel fix in place, **do not** suppress the whole non-finite bucket for these ops — group_norm/layer_norm/log_softmax non-finite is correctly inherited and the harness already matches; only the batch_norm `eps=0`/bad-var region needs the guard or the SKIP.

## Cited artifacts
- Jobs: `w0:1358` (lead, var=0/eps=0, finite inputs), `w0:143`, `w0:1539`, `w0:963` (batch_norm); `w10:1426 w10:246 w10:622` (group_norm); `w0:1569` (layer_norm, inherited via `log2`); `w0:1764` (log_softmax, inherited).
- Kernel: `pytorch_ref/executorch/kernels/portable/cpu/op_native_batch_norm.cpp:119,296`; ATen `pytorch_ref/aten/src/ATen/native/Normalization.cpp:104-112,194`.
- Isolation scripts + pasted output above.

## Replay

Self-contained minimal-export replay: [`replay_normalization-nonfinite.py`](replay_normalization-nonfinite.py).
Exports `torch.ops.aten._native_batch_norm_legit_no_training(x, None, None, mean, var, 0.0, 0.0)`
with finite `x=[[2.0]]`, `mean=[0.0]`, `var=[0.0]`, `eps=0` (so `(x-mean)=2 != 0`), runs it
on the portable ExecuTorch runtime vs eager, and asserts portable `±inf` where eager `nan`
(exits 0 iff reproduced). This is the `x-mean=2,var=0,eps=0,f32` isolation row.

```
$ cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_normalization-nonfinite.py 2>&1 \
    | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference"
x: [[2.0]] mean: [0.0] var: [0.0] eps=0
eager:    [nan]
portable: [inf]
  [0] portable=inf  eager=nan
REPLAY: bug REPRODUCED
```

Portable's unguarded `invstd = 1/sqrt(var+eps) = +inf` makes `(x-mean)*inf = +inf`;
eager's `InvStd` guard forces `invstd=0` when `var==0 && eps==0`, yielding `nan`.
