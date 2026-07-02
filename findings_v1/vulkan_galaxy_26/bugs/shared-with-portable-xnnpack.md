# Vulkan born-here root ops shared with portable / xnnpack findings

This is the "reuse, don't re-derive" section for the `vulkan_galaxy_26` run. Several
vulkan born-here clusters (first-divergence root ops from the on-device localizer,
`tmp/run_vulkan_galaxy_26/localized.jsonl`) are the **same root cause** already
documented for the portable and/or xnnpack runs. For each we confirmed the vulkan
device-recorded divergence (the localizer `detail` field, measured on the Samsung
Galaxy Adreno GPU) against eager and against the documented mechanism. Where the
mechanism matches, reference the existing finding instead of writing a new one.

Method: ran each example job's `g()` eager on host (`.venv/bin/python`) to get the
eager value, compared to the localizer's recorded device value, and matched the
arithmetic to the documented portable/xnnpack root cause. Did **not** touch the
device/broker.

## Summary table

| vulkan root op | born-here count | matching existing finding | confirmed same mechanism? | note |
|---|---:|---|---|---|
| `remainder` (delta) | ~75 | [`findings/portable/bugs/remainder-sign.md`](../../portable/bugs/remainder-sign.md) | **YES** | vulkan returns fmod (sign of dividend); eager returns floored remainder (sign of divisor). Identical to portable integer-path bug. |
| `bitwise_left_shift` (delta) | ~109 | [`findings/portable/bugs/bitwise-shift-negative.md`](../../portable/bugs/bitwise-shift-negative.md) | **YES** | negative/oversize shift count = UB; no clamp to eager's defined `0`. vulkan magic value is the same class (e.g. `2.306e+18 = 1<<61`). |
| `bitwise_right_shift` (delta) | ~23-39 | [`findings/portable/bugs/bitwise-shift-negative.md`](../../portable/bugs/bitwise-shift-negative.md) | **YES (same mechanism, platform-specific value)** | same unguarded shift UB; vulkan's wrapped garbage value differs from x86 portable (e.g. eager `-1` vs vulkan `-128`) because GLSL int is 32-bit and masks the count differently, but root cause = negative-shift UB vs eager's defined sign-saturation. |
| `select_scatter` (dtype) | ~57 | [`findings/portable/bugs/select_scatter-dtype.md`](../../portable/bugs/select_scatter-dtype.md) | **YES** | `to_edge()` decomposes `select_scatter` to `torch.where(mask, src, self)` which type-promotes; output dtype becomes `result_type(self,src)` not `self.dtype`. This is an **export-time** bug upstream of any kernel, so it is backend-agnostic and reproduces identically on vulkan (bool-vs-int64, int64-vs-float32 dtype flips). |
| `sum` (delta) | ~57 | [`findings/portable/bugs/sum-cast-order.md`](../../portable/bugs/sum-cast-order.md) | **YES (shared family)** | float-in/int-out reductions: accumulate-then-cast (and int64 overflow / fp16 saturation). vulkan device values show the same signature (`-9.223e+18` int64 overflow, `6.55e+04` fp16 max). **Note:** the int-out sum case is owned by a separate dedicated subagent for this run — cross-ref only. |
| `pow` (int64 overflow, delta/nonfinite) | ~41 delta / 70 nonfinite | [`findings/portable/bugs/prod-pow-int64-overflow.md`](../../portable/bugs/prod-pow-int64-overflow.md) | **PARTIAL** | the int64-overflow `pow` cases are UB-on-both-sides (flag-not-fix), shared with the portable prod/pow finding. **Caveat:** vulkan has no native 64-bit int, so the 70 `pow` *nonfinite* cases are a vulkan-specific int64-limitation / fp16-overflow surface — see the separate `vulkan-int64-truncation.md` / `vulkan-fp16-overflow.md` findings, not this overflow doc. |

## Cases that turned out NOT shared (vulkan-specific)

| vulkan root op | born-here count | why not shared |
|---|---:|---|
| `floor_divide` (delta) | ~81 | The documented portable finding ([`floor_divide.md`](../../portable/bugs/floor_divide.md)) is a **float by-zero ±inf** bug and explicitly proves portable's **integer** floor_divide is CORRECT (floors toward −∞, matches eager). On vulkan the integer path **truncates toward zero** instead of flooring — the exact bug the portable doc *disproves* for portable. Confirmed on host: eager `-1//4 = -1`, `-15//2 = -8` (floor); vulkan device gives `0` and `-7` (trunc). Same op, **different mechanism** → needs its own vulkan finding (integer floor-vs-trunc), not a cross-ref. |
| `index_put` (delta) | ~99 | No matching portable or xnnpack finding exists (`grep -rli index_put findings/` → none). 112/132 born-here index_put cases are `inherited:true`; the recorded device deltas are at **untouched** positions (e.g. w76:1620: eager `n1[0]=-0.5357` = the unchanged `mul` value at index 0; index_put only writes index 1, yet vulkan returns `0.6279` there). The eager values (`-0.4825`, `0.934`) are seeded leaf values, so the divergence is a vulkan-specific index_put / broadcast / accumulate copy issue, not a shared root cause. Vulkan-specific. |

## Evidence (example jobs, eager confirmed on host)

- **remainder** — `w10:1267`: `n0=bitwise_xor(L0,L1)=[-7,0,2]`; `remainder(n0,6)`: eager `[5,0,2]`, vulkan device `n1[0]=-1` = `fmod(-7,6)`. Matches portable remainder-sign (fmod vs floored). Also `remainder.Scalar(n2,-8)` (negative divisor) in the same graph.
- **bitwise_left_shift** — `w78:369`: `bitwise_left_shift.Tensor_out(n3, L1)` with `L1` int64 drawn from `[-4,5)` (negative shift counts). Device `e[0]=0 b[0]=2.306e+18` (`= 1<<61`), exactly the portable magic-number class.
- **bitwise_right_shift** — `w13:506`: `bitwise_right_shift.Tensor_Scalar_out(n4, -5)` (negative constant count). Device `e[0]=-1 b[0]=-128`; eager `x>>-5` sign-saturates to `-1`, vulkan returns wrapped `-128` (different 32-bit GLSL masking, same UB root cause).
- **select_scatter** — `w18:586` (`select_scatter(bool self, int64 src)` → device dtype `bool vs int64`), `w49:178` (`select_scatter(int64 self, float32 src)` → `int64 vs float32`). Both are the `where`-promotion decomposition flip.
- **floor_divide** — `w56:1050`: `floor_divide.default(L2, n0)` int64, device `e[0]=-1 b[0]=0`; `w16:1546`: `e[0]=-8 b[0]=-7`. Both = floor (eager) vs trunc (vulkan) on negative quotients.
- **index_put** — `w76:1620`, `w2:24`: divergence at positions index_put does not write; eager values are unchanged upstream leaf/mul values.

## Net

Shared (reuse the portable finding, no new vulkan derivation needed): **remainder,
bitwise_left_shift, bitwise_right_shift, select_scatter, sum (family), pow
int64-overflow (partial)**. Of these, `select_scatter` and `sum` are export-level /
backend-agnostic (reproduce identically regardless of backend); the shift and
remainder bugs are the same *mechanism* with platform-specific garbage values.

Vulkan-specific (need their own findings, NOT a cross-ref): **floor_divide**
(integer trunc-instead-of-floor — the portable doc actively disproves this for
portable) and **index_put** (no portable/xnnpack analog; positional copy/broadcast
divergence). The `pow` *nonfinite* cluster is also vulkan-specific (int64/fp16
limitation), distinct from the shared int64-overflow `pow` delta cluster.
