# Ruled-out suspects (not XNNPACK-delegate bugs)

All non-OK outcomes below were removed by the `analysis_single.md` step-3 filters. They are real
divergences, but not bugs in the **XNNPACK delegate** (the backend under test).

## Filter 3a — portable-fallback (`delegated.ops = 0`)
XNNPACK only delegates a subset of ops; everything else falls back to the **portable** CPU kernels.
A divergence on a portable-fallback job is a portable-kernel or reference issue, **not** an XNNPACK
delegate bug. In this run:

- **1,036 of 1,128 mismatches** were portable-fallback (`ops=0`). Only the 92 `sqrt.out` mismatches
  ran on the XNNPACK kernel.
- **All 42 crashes** were portable-fallback (`narrow_copy`, `unfold_copy` neg-dim / 0-D; integer
  divide SIGFPE) — attributed to the portable kernels, not XNNPACK.
- **2,204 of 2,311 skips** were portable-kernel guards / coverage gaps (`_fft_r2c`, `arange`,
  `_pdist_forward`, `scatter*`, `linear`, `copy`, `expand_copy` implicit, `convolution`, …). Only 107
  skips were genuine XNNPACK-runtime errors (`XNNCompiler`/`XNNExecutor`) — see the confirmed SKIP
  bugs in [../bugs/](../bugs/).

Top portable-fallback mismatch ops (ruled out): `topk.values` (153), `bitwise_left_shift.*` (543
across variants), `remainder.*` (159), `native_group_norm` (26), `_native_batch_norm_legit` (41),
`min.dim`/`max.dim` (48), `prod.int_out` (16), `fmod` (27). These reproduce the *same* portable
kernels regardless of backend and belong to a portable analysis, not this one.

## Filter 3d — reference-side INT64-sentinel confounds
~260 portable-fallback mismatches had `|delta| ≈ 9.223e18` (= `2^63−1`, `INT64_MAX`). These are
int64 CPU-fallback sentinels in the **reference** (cumsum/argmin/argmax-type int64 ops), not device
garbage — they mismatch alone by construction. Confound, not a device bug.

## Note
None of these were investigated further for XNNPACK. If a portable analysis is wanted, they are the
starting worklist there.
