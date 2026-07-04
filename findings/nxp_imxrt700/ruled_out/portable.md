# Ruled out — portable-fallback (not Neutron) — filter 3a

Neutron delegates only `relu` and `abs`; **98% of the corpus runs on the portable CPU kernels** in
`nxp_executor_runner`. Divergences/crashes there are host-runner or portable-kernel behaviour, NOT
bugs in the Neutron delegate under test. Excluded from the finding.

## Portable CRASHes (all `ops=0`, `nxp_runner rc=255` — runner error, not a native abort)
| count | operator |
|--:|---|
| 760 | `native_group_norm` |
| 441 | `stack` |
| 441 | `_fft_r2c` |
| 211 | `_pdist_forward` |
| 149 | `copy` |
| 136 | `expand_copy` |
| 31 | `convolution` |
| 7 | `narrow_copy` |
| others | `_adaptive_avg_pool2d`, `_native_batch_norm_legit_no_training`, `unfold_copy` |

`rc=255` = the host runner failed to load/execute the portable graph (unsupported portable op or a
runner-side error). A portable/runner coverage gap; worth a portable follow-up, not a Neutron bug.

## Portable MISMATCHes (all `ops=0`)
~3,251 portable-kernel divergences (e.g. `detach_copy`, `t_copy`, `unbind_copy`), many with
`|delta|=255` — uint8 dtype/quant isolation artifacts (§3c), not Neutron.
