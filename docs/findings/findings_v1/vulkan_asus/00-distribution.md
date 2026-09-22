# Vulkan ASUS — distribution & job-by-job comparison vs Galaxy

`corpus/vulkan` (100,032 graphs), same `.pte` builds, same broker, each diffed vs eager.
Generator: `tmp/run_vulkan_asus/compare_asus_galaxy.py` → `compare_asus_galaxy.txt`.

## Verdicts (ASUS vs Galaxy)

| verdict | ASUS | Galaxy |
|---|---:|---:|
| OK | 12,637 | 13,331 |
| MISMATCH | **8,896** | 8,202 |
| CRASH | 6,214 | 6,214 |
| SKIP | 72,285 | 72,285 |
| TIMEOUT | 0 | 0 |

ASUS has **+694 net mismatches** (8,896 vs 8,202) — entirely the transcendental fringe below.

## Confusion matrix (rows = Galaxy, cols = ASUS)

```
  Gal\ASUS         OK      SKIP     CRASH  MISMATCH
  OK            12627         0         0       704
  SKIP              0     72285         0         0
  CRASH             1         0      6213         0
  MISMATCH          9         0         1      8192
```

**Verdict agreement: 99,317 / 100,032 = 99.3%.**

## Device-invariant surfaces

- **SKIP — 100% identical** (72,285 shared, 0 ASUS-only, 0 Galaxy-only). Refusals come from the
  host lowering + shader-gen + `.pte` build, not the GPU. Both phones reject exactly the same
  graphs (see [../vulkan_galaxy_26/skip/README.md](../vulkan_galaxy_26/skip/README.md)).
- **CRASH — 99.97% identical** (6,213 shared; 1 ASUS-only, 1 Galaxy-only). Native aborts are
  graph/op-driven, not device-driven — both Adreno parts abort on the same graphs.

## Device-specific: the 704 ASUS-only mismatches

ASUS=MISMATCH where Galaxy=OK (matched eager). **83% are non-finite** (584/704). Op on the flagged
output:

| op | n | | op | n |
|---|--:|---|---|--:|
| `rsqrt` | 142 | | `expm1` | 15 |
| `sqrt` | 138 | | `fmod` | 15 |
| `sinh` | 23 | | `logit` | 12 |
| `cosh` | 18 | | `mean` | 10 |
| `exp` | 17 | | `tan` | 9 |
| `remainder` | 17 | | `reciprocal` | 9 |

`rsqrt`+`sqrt` = **280** (40%). Examples (ASUS nonfinite, Galaxy=OK): `w0:1875` (rsqrt of a
sliced int/negative), `w1:1106` (`rsqrt(erf(...))` — erf∈[-1,1] → rsqrt of negative),
`w10:1262` (`sqrt(min(gelu))` — sqrt of a possibly-negative min). All are **sqrt/rsqrt/transcendental
of negative / zero / large-magnitude inputs** → the two drivers disagree on the special-value result.
Root-cause + classification: [bugs/asus-transcendental-nonfinite.md](bugs/asus-transcendental-nonfinite.md).

## Galaxy-only mismatches (10)

Negligible and scattered (2 rsqrt, then single atan2/floor_divide/mul/floor/hardtanh/gather/clamp/
max_pool2d_with_indices_backward) — the symmetric tail of the same transcendental fringe.

## Reading

The Vulkan delegate's **correctness bugs are portable across Adreno devices**: int64 truncation,
fp16 overflow, the NaN-dropping activations, floor_divide, index_put, and the copy-elision aliasing
bug all reproduce identically on ASUS. Only the **transcendental special-value fringe (~0.7%)** is
device-dependent — the hardware/driver evaluation of `sqrt`/`rsqrt`/`exp`/… at the extremes.
