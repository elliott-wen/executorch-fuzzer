# CoreML native-abort crashes — a family of delegated ops that hard-abort the runtime

- **Failure mode:** CRASH (`executor died (native abort)` — no catchable error message)
- **Root cause:** operator kernel / lowering (delegated to Core ML; `delegated.ops≥1`)
- **Mechanism:** native abort. The delegate aborts the process loading/running the op instead of
  rejecting the offending form with a catchable `Check failed`. **The missing guard is the bug.**
- **Repro:** [repro_crash-native-abort.py](repro_crash-native-abort.py)

## Verification & the collateral caveat
The bulk run logged **268** CRASH rows, but ~48% were **collateral**: when one graph natively aborts
a worker, other in-flight graphs on that same worker are also logged `native abort` though they are
innocent. Re-running every crash candidate **once each at `--window 1`** (a native abort then only
takes down its own job) plus an **N=5 same-job gate** on the borderline ones separated real from
collateral. Only the operators below reproduced.

## Confirmed native-abort operators (device-verified)
Reproduction = fraction of that op's distinct crashing samples that aborted again on the clean
`window=1` re-run (or N/5 on the same job for the gated ones).

| operator | reproduction | notes |
|----------|--------------|-------|
| `unfold_copy`                                 | 25/25 | aborts on the sliding-window copy |
| `prod.int_out`                                | 17/17 | also has a uint8 saturation mismatch form |
| `diagonal_copy`                               | 17/17 | |
| `scatter.value_out`                           | 16/16 | |
| `max_pool2d_with_indices_backward.grad_input` | 14/14 | also a WRONG-VALUE form — see [max_pool2d-backward.md](max_pool2d-backward.md) |
| `linear.out`                                  | 5/5   | |
| `mean.out`                                    | 3/3   | (the `mean` int form is a separate mismatch — see [mean-int.md](mean-int.md)) |
| `cumsum.out`                                  | 5/5 (gate) | |
| `logical_or.out`                              | 5/5 (gate) | form-specific |
| `mm.out`                                      | 5/5 (gate) | form-specific |
| `t_copy`                                      | 5/5 (gate) | form-specific |
| `tril.out`                                    | 5/5 (gate) | form-specific |
| `alias_copy`                                  | 5/5 (gate) | form-specific |
| `view_copy`                                   | 4/5 (gate) | form-specific (1 sample among many) |

Native aborts carry **no catchable message** (only `executor died (native abort)`); the crashing
form's dtype/shape/args are recorded via the corpus job id. See the CRASH table in
[../skips.md](../skips.md) for the per-op crashing forms.

## Ruled out as collateral (re-ran OK / SKIP, not real crashes)
`mul.out` (0/5), `abs`, `bmm.out`, `native_layer_norm`, `minimum.out`, `sign.out`, `eq.Tensor_out`,
`bitwise_xor.Scalar_out`, `split_copy.Tensor`, `min.dim_min`, `max.dim_max`, `pow.Scalar_out`,
`relu`, `where.self`, `sigmoid.out`, and ~25 more — all in
[../ruled_out/README.md](../ruled_out/README.md). `avg_pool2d.out`'s "crashes" were mostly SKIP
(static-tensor resize).
