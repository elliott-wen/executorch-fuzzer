# Ruled out: portable-fallback non-OK (`ops=0`) — not the mtk delegate (3a)

Only 65 of 171 corpus ops reach the Neuron delegate; the other 106 always fall back to the portable
CPU kernel (`ops=0`). Divergence there is a portable-kernel or reference issue, **not a bug in the mtk
backend under test.** Recorded separately:

- **MISMATCH 1,255** — top ops: `bitwise_left_shift.*` (740 across variants), `remainder.*` (~200),
  `grid_sampler_2d` (81), `prod`/`prod.int_out` (74), `_native_batch_norm_legit.no_stats` (46),
  `fmod.Scalar` (34), `bitwise_right_shift.*` (34), `sum.IntList_out` (30), `var_mean.correction` (3).
  (`bitwise_left_shift` dominating suggests a portable int-shift/overflow or reference quirk — a
  portable-side lead, not mtk.)
- **SKIP 1,011** — portable-runtime coverage gaps: `_fft_r2c` (442, "Operator missing"),
  `_pdist_forward` (348), `scatter*`/`gather` (213), etc. See `../skips.md`.
- **CRASH 11** — negligible portable-side aborts.

None are attributed to the MediaTek backend.
