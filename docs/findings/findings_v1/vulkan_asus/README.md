# Vulkan (ASUS, on-device) differential-fuzz — findings

Second on-device Vulkan run: `corpus/vulkan` (100,032 graphs) executed on an **ASUS phone**
(Adreno GPU, ExecuTorch **Vulkan** delegate), jobs dispatched from the host broker over a
`rathole` tunnel, each diffed vs eager PyTorch. Run **compared job-by-job against the
[Galaxy run](../vulkan_galaxy_26/README.md)** on the same corpus to isolate what's device-specific.

## TL;DR — is there any difference?

**Almost none.** Verdict agreement is **99.3%** (99,317 / 100,032). The two Adreno phones behave
identically except for a narrow transcendental fringe:

| surface | difference |
|---|---|
| **SKIP** (72,285) | **0 device-specific** — byte-identical on both phones. Refusals are decided by host lowering + shader-gen, not the GPU. |
| **CRASH** (~6,214) | **6,213 shared**, 1 ASUS-only, 1 Galaxy-only (noise). Native aborts reproduce on both Adreno parts. |
| **MISMATCH** | **8,192 shared** + **704 ASUS-only** + 10 Galaxy-only. The shared 8,192 are the documented [Galaxy bugs](../vulkan_galaxy_26/README.md) (int64, fp16, the real kernel bugs) — all device-invariant. |

So: **every correctness bug, crash, and skip found on Galaxy reproduces on ASUS.** The only
device-specific behaviour is **704 ASUS-only mismatches (~0.7% of the corpus)**, 83% non-finite,
dominated by transcendental ops.

## The one ASUS-specific finding → [bugs/asus-transcendental-nonfinite.md](bugs/asus-transcendental-nonfinite.md)

**ASUS's `sqrt`/`rsqrt`/`exp`/`sinh`/… emit NaN/Inf at positions where eager (and Galaxy) stay
finite** (or vice-versa) — a GPU/driver-level special-value difference on edge inputs (negatives,
zeros, large magnitudes). `rsqrt`(142)+`sqrt`(138) = 280 of the 704. Mirrors the NEON-vs-SSE
`f32-vlog` special-value finding from the [xnnpack-arm run](../xnnpack-arm/bugs/neon-vlog-special-values.md),
one layer down (two GPU drivers running identical SPIR-V).

## Everything else → see the Galaxy findings (no new work)

The shared 8,192 mismatches, all crashes, and all skips are identical in cause to Galaxy. Don't
re-derive — reuse:
- [vulkan_galaxy_26/README.md](../vulkan_galaxy_26/README.md) — the bug catalog (int64 truncation,
  fp16 overflow, tanh/sign/log_softmax NaN, floor_divide, index_put, copy-elision aliasing, …).
- [vulkan_galaxy_26/skip/README.md](../vulkan_galaxy_26/skip/README.md) — the 72k skips (identical here).
- [vulkan_galaxy_26/crash/README.md](../vulkan_galaxy_26/crash/README.md) +
  [bisect-localizer.md](../vulkan_galaxy_26/bisect-localizer.md) — crash culprits (identical here).

## Distribution & comparison

Full verdict table, confusion matrix, and per-op device-specific breakdown:
[00-distribution.md](00-distribution.md). Raw data + script: `tmp/run_vulkan_asus/`
(`skip_reasons_asus.tsv`, `compare_asus_galaxy.py` → `compare_asus_galaxy.txt`).
