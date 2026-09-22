# VGF confirmed bugs — index (corpus_v4)

Each bug is labeled by **failure mode** (mismatch / crash / skip) and, for mismatch, **root cause**
(operator / graph-optimization). Device-verified; see `../README.md` for the full synthesis.

## Mismatch · graph-optimization
- **`repro_dropped_store.py`** — ⭐ **ZEROED dropped store** under the multi-output memory plan.
  239 device-verified cases. A VGF-delegated op is correct returned ALONE but written **all-zeros**
  when a **view/copy-family sibling** (squeeze/clone/alias_copy/transpose/permute/view/expand/
  split_with_sizes_copy) is co-returned. Target op arbitrary; trigger = view/copy op. Memory-planner
  copy-elision/aliasing bug (VGF analog of `bugs/vulkan-copy-elision-aliasing.md`).
- COMPOSITIONAL fp16 number-format (275 VALUE cases) + NONFINITE/ALIAS/SCALE/SHAPE tails — see
  `../graphopt_all.tsv`; same A/B harness (`tmp/vgf_go_deep.py`).

## Mismatch · operator (VGF-delegate, N=5 deterministic)
- **`repro_operator_bugs.py`** — 21 confirmed ops. Cleanest: `native_group_norm` (clamps to ±1),
  `cumsum` (int → fp16 garbage), `sum` (1→0), `floor_divide` (sign), `log1p` (nan→finite). Full gate
  in `../opgate.tsv`.

## Crash · (ruled out — portable, not VGF)
- 109 native aborts, **all `delegated_ops=0`** → portable-kernel crash cluster (`narrow_copy`/
  `unfold_copy` in composition). Not a VGF-delegate bug. See `../crash_triage.tsv`.

## Skip · runtime coverage gaps
- 90% VGF-delegate **execution failures** (emulation layer can't dispatch the TOSA blob); 10%
  missing portable kernels (`aten::linear.out`, `aten::_fft_r2c.out`). See `../skip_triage.tsv`.

## Ruled-out suspects (investigated, NOT filed as VGF bugs)
- Portable-fallback OPERATOR confounds: `bitwise_left_shift`, `fmod`, `_native_batch_norm_legit`,
  `bitwise_right_shift`, `_upsample_bilinear2d_aa`, `maximum`, `var_mean` (`delegated_ops=0`).
- Flaky (0/5 on N=5 re-run): `acos`, `addmm`, `div`, `logit`, `sinh`, `_softmax`.
- Non-finite mismatches that are VGF/TOSA **saturation** of a torch-propagated nan/inf (not sink-op bugs).
