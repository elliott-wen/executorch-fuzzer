# Build-time coverage gap: linear / linear.out / _fft_r2c (rc=5, NOT device crashes)

Graphs whose lowered .pte references `aten::linear.out` (linear that stayed portable — NO-CM,
unquantized) or `_fft_r2c` fail the per-pte semihosting runner BUILD, so they never execute on
the FVP. The client reports rc=5, logged as CRASH but split out as CRASH_INFRA (filter: rigor
reminder — infra/build failures are not device crashes).

Root cause (reproduced, deterministic): the executorch arm portable_ops_lib codegen emits
  arm_portable_ops_lib/RegisterCodegenUnboxedKernelsEverything.cpp:15:10: error: empty filename in #include
i.e. the selected op has NO portable kernel source to include. `aten::linear.out` has no portable
kernel (linear is normally decomposed to addmm/mm), so selecting it alone produces an empty #include.

Attribution: executorch portable-codegen / missing-portable-kernel + fvp_runner per-pte linking —
NOT a cortex-m backend device bug. These ops get NO device verdict under this harness → coverage gap.
