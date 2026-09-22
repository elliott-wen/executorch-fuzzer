# Portable corpus — differential-fuzz run: result distribution

**Run:** portable-lowered `.pte` corpus executed in-process on the ExecuTorch host runtime via `xnnpack_client`
(the in-process runtime runs portable- and xnnpack-lowered programs alike). Reference = eager PyTorch; the
feeder diffs returned output tensors against eager and tallies a verdict per graph.

- **Corpus:** `corpus/portable/` — 117,288 graphs fed (window=64, per-job timeout 30s)
- **Result TSV:** `tmp/run_portable/skip_reasons_portable.tsv` (failing rows only; OK rows not logged)
- **Feeder log:** `tmp/run_portable/feeder_final.log`

## Top-level verdicts

| verdict | count | share | meaning |
|---|---:|---:|---|
| OK | 63,480 | 54.1% | outputs matched eager within tolerance |
| MISMATCH | 10,066 | 8.6% | ran but outputs differ from eager |
| CRASH | 5,996 | 5.1% | native abort took the executor fork down |
| SKIP | 37,746 | 32.2% | portable kernel refused the call (Check/Unhandled — graceful error) |
| TIMEOUT | 0 | 0.0% | no answer within deadline |
| **total** | **117,288** | 100% | |

> Of programs that *ran to completion* (OK+MISMATCH = 73,546), 13.7% mismatched eager. SKIP (32.2%) are graceful 'kernel not implemented for these args' refusals, not bugs.

## SKIP distribution — portable kernel coverage gaps

Graceful refusals: the portable kernel ran a runtime `Check`/dtype guard and returned an error (no crash).
Clustered by `(operator, guard that fired)`.

| count | operator | guard / reason |
|---:|---|---|
| 10,023 | `aten::_conj_physical.out` | Unhandled dtype |
| 4,821 | `aten::expand_copy.out` | expand: implicit==true unimplemented |
| 4,754 | `aten::copy.out` | Check failed: non_blocking == false |
| 3,839 | `aten::native_group_norm.out` | Check failed: in.size(0 |
| 2,872 | `aten::_log_softmax.out` | Check failed: false |
| 2,865 | `aten::_pdist_forward.out` | Check failed: static_cast<size_t>(t.dim( |
| 1,693 | `aten::cumsum.out` | Check failed: dim >= -upper_bound && dim < upper_bound |
| 1,170 | `aten::_fft_r2c.out` | Check failed: onesided |
| 930 | `aten::add.Scalar_out` | Check failed: (common_type == a_type && check_alpha_type(utils |
| 785 | `aten::_adaptive_avg_pool2d.out` | Check failed: output_size[0] > 0 && output_size[1] > 0 |
| 618 | `aten::_native_batch_norm_legit_no_training.out` | Check failed: a.scalar_type( |
| 613 | `aten::native_group_norm.out` | Check failed: in.size(1 |
| 592 | `aten::convolution.out` | Check failed: in.dim( |
| 554 | `aten::arange.start_out` | Check failed: utils::extract_scalar(start, &d_start |
| 375 | `aten::_fft_r2c.out` | Check failed: all_contiguous || all_channels_last |
| 214 | `aten::scatter.value_out` | Check failed: index.scalar_type( |
| 201 | `aten::_pdist_forward.out` | Unhandled dtype |
| 114 | `aten::scatter.src_out` | Check failed: index.scalar_type( |
| 101 | `aten::scatter_add.out` | Check failed: index.scalar_type( |
| 58 | `aten::permute_copy.out` | Unhandled dtype |
| 51 | `aten::div.out` | Check failed: error == Error::Ok |
| 39 | `aten::sub.out` | Check failed: error == Error::Ok |
| 38 | `aten::arange.start_out` | Check failed: utils::extract_scalar(end, &d_end |
| 27 | `aten::remainder.Scalar_out` | Check failed: !(executorch::runtime::isIntegralType(common_typ |
| 26 | `aten::fmod.Scalar_out` | Check failed: !(executorch::runtime::isIntegralType(common_typ |
| 24 | `aten::bitwise_or.Scalar_out` | Unhandled dtype |
| 23 | `aten::pow.Tensor_Scalar_out` | Check failed: (canCast(common_type, out.scalar_type( |
| 21 | `aten::copy.out` | Check failed: a.scalar_type( |
| 20 | `aten::bitwise_left_shift.Tensor_Scalar_out` | Unhandled dtype |
| 20 | `aten::bitwise_right_shift.Tensor_Scalar_out` | Unhandled dtype |

## MISMATCH distribution — ran but diverged from eager

| count | cluster | interpretation |
|---:|---|---|
| 2,937 | `non-finite` | nan/inf appear in different positions (overflow/invalid-domain handled differently) |
| 1,866 | `delta:1-1e6:strict` | moderate exact diff (int/rounding ops) |
| 1,336 | `delta:>=1e15:strict` | huge magnitude diff, exact-compare ops (int/bitwise) — likely int overflow/wrap divergence |
| 1,090 | `delta:1-1e6:loose` | moderate float diff |
| 657 | `delta:>=1e15:loose` | huge magnitude diff under loose tol — float overflow divergence |
| 551 | `dtype:torch.int64_vs_torch.float32` | dtype promotion differs (int64 vs fp32) |
| 536 | `dtype:torch.int64_vs_torch.float16` | dtype promotion differs (int64 vs fp16) |
| 485 | `shape` | output shape differs from eager |
| 430 | `delta:<1:loose` | small float precision diff |
| 125 | `dtype:torch.bool_vs_torch.int64` | bool vs int64 result dtype |
| 16 | `dtype:torch.float16_vs_torch.float32` | fp16 vs fp32 result dtype |
| 12 | `delta:1e6-1e15:strict` | large exact diff |
| 8 | `delta:1e6-1e15:loose` | large float diff |
| 7 | `dtype:torch.bool_vs_torch.float16` | — |
| 6 | `dtype:torch.float32_vs_torch.float16` | — |
| 4 | `dtype:torch.bool_vs_torch.float32` | — |

**Ops most often on a mismatching output node (starred = compared output):**

| count | op |  | count | op |
|---:|---|---|---:|---|
| 1,945 | `remainder` |  | 1,095 | `min` |
| 1,488 | `floor_divide` |  | 1,069 | `logical_and` |
| 1,417 | `div` |  | 1,063 | `max` |
| 1,400 | `bitwise_left_shift` |  | 1,051 | `logical_not` |
| 1,389 | `select_scatter` |  | 1,046 | `logical_xor` |
| 1,366 | `narrow_copy` |  | 1,017 | `squeeze_copy` |
| 1,322 | `fmod` |  | 1,005 | `gt` |
| 1,306 | `mean` |  | 992 | `le` |
| 1,281 | `pow` |  | 984 | `mul` |
| 1,262 | `prod` |  | 979 | `ge` |

## CRASH distribution — native aborts

The reason is always *“executor died (native abort)”* — the abort gives no op, so we attribute it by **op
enrichment**: how much more often an op appears in crashing graphs (5,996) than in non-crashing ones (47,812).

| op | in crashes | p(crash) | p(other) | **lift** |
|---|---:|---:|---:|---:|
| `unfold_copy` | 2,116 | 0.353 | 0.090 | 3.91 |
| `narrow_copy` | 4,334 | 0.723 | 0.209 | 3.45 |
| `ones` | 52 | 0.009 | 0.005 | 1.77 |
| `zeros` | 59 | 0.010 | 0.006 | 1.67 |
| `pad` | 1,105 | 0.184 | 0.114 | 1.61 |
| `t_copy` | 535 | 0.089 | 0.067 | 1.32 |
| `any` | 768 | 0.128 | 0.105 | 1.22 |
| `asinh` | 675 | 0.113 | 0.098 | 1.14 |
| `full` | 40 | 0.007 | 0.006 | 1.13 |
| `atanh` | 640 | 0.107 | 0.096 | 1.11 |
| `linear` | 750 | 0.125 | 0.112 | 1.11 |
| `bitwise_xor` | 694 | 0.116 | 0.104 | 1.11 |
| `sin` | 628 | 0.105 | 0.095 | 1.10 |
| `max` | 1,260 | 0.210 | 0.192 | 1.10 |
| `floor` | 580 | 0.097 | 0.088 | 1.09 |

**Crash clusters** (each crash assigned to its highest-lift suspect op present; suspects = lift≥1.5 & ≥30 crashes):

| count | suspect op |
|---:|---|
| 3,758 | `narrow_copy` |
| 2,116 | `unfold_copy` |
| 83 | `other` |
| 16 | `zeros` |
| 14 | `pad` |
| 9 | `ones` |
