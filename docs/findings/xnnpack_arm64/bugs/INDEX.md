# Confirmed XNNPACK-delegate bugs (arm64 phone) — index

Device-verified on the phone. All delegated (`ops≥1`).

| operator | mode | arch | count | eager → device | file |
|---|---|---|---:|---|---|
| `log` | MISMATCH | arm64-only | 180 | `log(inf)`: inf → 88.38 | [xnnpack-log-saturate.md](xnnpack-log-saturate.md) |
| `logit` | MISMATCH | arm64-only | 187 | `+inf` → 88.38 | [xnnpack-logit-saturate.md](xnnpack-logit-saturate.md) |
| `gelu` | MISMATCH | shared | 92 | `gelu(inf)`: NaN → inf | [xnnpack-gelu-nonfinite.md](xnnpack-gelu-nonfinite.md) |
| `max_pool2d_…_backward`, `unfold_copy`, `narrow_copy` | CRASH | shared | ~60 | native abort | [../skips.md](../skips.md) |

Arch differential vs x86 and the flaky-worker correction: [../README.md](../README.md).
