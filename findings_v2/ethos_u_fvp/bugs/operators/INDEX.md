# Operator bugs — Ethos-U55 on Arm Corstone-300 FVP (int8 / Vela)

Device-verified by single-op isolation with EXACT baked inputs, **then gated by 5× stability
re-verification** (`../../REVERIFICATION.md`). A single GENUINE run is not a confirmed bug on an
int8 backend — near-tolerance divergences are non-deterministic.

**5 STABLE (confirmed) · 8 intermittent · 13 dropped as flaky (0/5).**

## Root causes of the 5 confirmed bugs (3 distinct)
- **uint8-range value → signed-int8 overflow (1 root cause, 2 ops):** `pow.Scalar_out` (250→0) and
  `prod.int_out` (250/251/253→0). Vela assigns the output a **signed int8** tensor (scale ≈ 1.0, zp 0,
  clip [−128,127]) even though the values are uint8-range (0–255), so anything ≥128 overflows to 0.
  Confirmed from the Vela command stream (`ACTIVATION_TYPE_SIGNED`, `ACTIVATION_MIN=0xFF80`,
  `ACTIVATION_MAX=127`). One underlying defect — see `pow_Scalar_out.md`, `prod_int_out.md`.
- **spurious NaN (2 ops):** `prod.default` (0 → NaN), `slice_scatter.default` (scattered element → NaN)
  — the op manufactures a non-finite from finite inputs (bad zero-factor reduction / uninitialised
  scatter store).
- **off-by-one rounding (1 op):** `floor_divide.default` — one element floors the wrong direction (Δ=1).

The graph-optimization bug (sibling-dependent requantization scale error) is separate — see
`../graphopt-requant-scale.md`.

## Confirmed operator bugs (5/5 reproduce) (5)

| op | reproduced | note | doc |
|---|---|---|---|
| `floor_divide.default` | 5/5 |  | [floor_divide_default](floor_divide_default.md) |
| `pow.Scalar_out` | 5/5 |  | [pow_Scalar_out](pow_Scalar_out.md) |
| `prod.default` | 5/5 |  | [prod_default](prod_default.md) |
| `prod.int_out` | 5/5 |  | [prod_int_out](prod_int_out.md) |
| `slice_scatter.default` | 5/5 |  | [slice_scatter_default](slice_scatter_default.md) |

## Intermittent / non-deterministic kernels (1–4/5) (8)

| op | reproduced | note | doc |
|---|---|---|---|
| `_upsample_bilinear2d_aa.out` | 1/5 | dtype (likely harness artifact) | [_upsample_bilinear2d_aa_out](_upsample_bilinear2d_aa_out.md) |
| `bitwise_left_shift.Tensor_out` | 3/5 |  | [bitwise_left_shift_Tensor_out](bitwise_left_shift_Tensor_out.md) |
| `bitwise_right_shift.Tensor_out` | 2/5 |  | [bitwise_right_shift_Tensor_out](bitwise_right_shift_Tensor_out.md) |
| `clamp.Tensor_out` | 1/5 | dtype (likely harness artifact) | [clamp_Tensor_out](clamp_Tensor_out.md) |
| `fill.Scalar` | 1/5 | dtype (likely harness artifact) | [fill_Scalar](fill_Scalar.md) |
| `logit.default` | 2/5 |  | [logit_default](logit_default.md) |
| `mean.default` | 1/5 |  | [mean_default](mean_default.md) |
| `rsub.Scalar` | 1/5 |  | [rsub_Scalar](rsub_Scalar.md) |

## Dropped — did not reproduce (0/5, near-tolerance noise) (13)

| op | reproduced | note | doc |
|---|---|---|---|
| `_adaptive_avg_pool2d.default` | 0/5 |  | [_adaptive_avg_pool2d_default](_adaptive_avg_pool2d_default.md) |
| `addmm.out` | 0/5 |  | [addmm_out](addmm_out.md) |
| `alias_copy.default` | 0/5 |  | [alias_copy_default](alias_copy_default.md) |
| `any.dims_out` | 0/5 |  | [any_dims_out](any_dims_out.md) |
| `bitwise_right_shift.Tensor_Scalar` | 0/5 |  | [bitwise_right_shift_Tensor_Scalar](bitwise_right_shift_Tensor_Scalar.md) |
| `bitwise_right_shift.Tensor_Scalar_out` | 0/5 |  | [bitwise_right_shift_Tensor_Scalar_out](bitwise_right_shift_Tensor_Scalar_out.md) |
| `div.Scalar` | 0/5 |  | [div_Scalar](div_Scalar.md) |
| `elu.out` | 0/5 |  | [elu_out](elu_out.md) |
| `eq.Tensor_out` | 0/5 |  | [eq_Tensor_out](eq_Tensor_out.md) |
| `floor_divide.out` | 0/5 |  | [floor_divide_out](floor_divide_out.md) |
| `mean.out` | 0/5 |  | [mean_out](mean_out.md) |
| `native_group_norm.default` | 0/5 |  | [native_group_norm_default](native_group_norm_default.md) |
| `native_layer_norm.default` | 0/5 |  | [native_layer_norm_default](native_layer_norm_default.md) |
