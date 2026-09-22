# Ruled-out suspects (portable run)

# Ruled-out suspects (portable run — 87,921 graphs)

## `bitwise_left_shift.*` — integer shift overflow / reference confound (736)
`.Tensor_out` (335), `.Tensor_Scalar` (202), `.Tensor_Scalar_out` (199). Dominated by
`|delta| ≈ 9.223e18` (`2^63−1`, `INT64_MAX`) — reference-side int64 sentinels (filter 3d), plus
shift-by-large-count which is undefined behavior in C (portable and ATen make different UB choices).
Not a clean portable kernel bug; excluded.

## `any.*` — portable CORRECT, eager reference is the anomaly (9)
`any.dims_out` / `any.all_out` on a non-finite input: portable returns `True` (`1.0`), matching
`torch.any([inf,inf]) == True`. The **stored eager reference** returned `False` (`0.0`) on the
`out=uint8` form — a reference-side artifact, not a portable bug. Device-verified: the portable answer
is the mathematically correct one. Excluded (would be a false positive if filed).

## `pow.*` (8), `sum.IntList_out` (5), `var_mean.correction` (1) — int64 / near-tolerance confounds
`pow.Scalar_out`/`pow.Tensor_Tensor_out` and `sum.IntList_out` are dominated by `2^63`/`2^31` int64
sentinels; `var_mean` is a single near-tolerance sample. Reference/UB confounds, not device bugs.

None is filed as a portable bug.
