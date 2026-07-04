# QNN HTP has no IEEE non-finite semantics — inf/nan produce wrong finite values

- **Mode:** MISMATCH (non-finite) · **Root cause:** operator kernels (delegated) · **Occurrences:** ~5,374 (the dominant class)
- Surfaced by the non-finite leaf injection; device-verified on the QNN HTP x86 emulator.

The Qualcomm HTP fp16 path does **not** implement IEEE `inf`/`nan` behavior — every transcendental /
activation kernel returns a wrong **finite** value (or the wrong non-finite) instead of propagating.
Device-verified examples:

| op | eager (ATen) | QNN HTP |
|---|---|---|
| `log(inf)` | inf | **-0.000489** |
| `exp(inf)` | inf | **131008** (overflow to a huge finite) |
| `cos(nan)` | nan | **1.0** |
| `sin(nan)` | nan | wrong |
| `atan(inf)` | π/2 (1.5703) | **nan** |
| `elu(inf)` | inf | **-65472** (sign flip + saturate) |
| `sqrt(neg)` | nan | **-131008** |
| `logit`, `log`, `exp`, `cos`, `sin`, `atan`, `rsqrt` | — | non-finite mishandled |

Affected ops (delegated non-finite mismatches): `logit` (649), `index_put` (318), `elu` (296),
`sqrt` (282), `log` (277), `exp` (244), `cos` (238), `linear` (228), `index` (212), `sin` (204),
`mul` (195), … The recurring signature is `inf → ±131008 / ±65472 / ~0` and `nan → a plausible-but-wrong
finite`. A whole-backend gap: the HTP fp16 kernels never see IEEE non-finites in normal models, so the
domain is unimplemented.
