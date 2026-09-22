# Ruled out — portable-fallback divergences (step 3a: not OpenVINO)

These non-OK outcomes occurred on jobs where `delegated.ops == 0` — the OpenVINO delegate **never
ran the op**; the device executed the **portable CPU kernel**. A divergence there is a portable-kernel
or reference issue, **not a bug in the backend under test**. Removed before attribution (step 3a).

## Portable-fallback MISMATCH (583 jobs, 6 operators)
| count | operator |
|--:|---|
| 199 | `bitwise_left_shift.Tensor_Scalar` |
| 133 | `bitwise_left_shift.Tensor_out` (portable instances) |
| 82  | `remainder.Scalar` |
| 75  | `remainder.Tensor_out` (portable instances) |
| 72  | `bitwise_left_shift.Tensor_Scalar_out` (portable instances) |
| 22  | `remainder.Scalar_out` (portable instances) |

Note: the same op names also appear as **delegated** instances (kept as findings) — the op sometimes
partitions to OpenVINO and sometimes falls back. Only the `ops≥1` instances are OpenVINO bugs.

## Portable-fallback CRASH (11 jobs, 2 operators)
| count | operator |
|--:|---|
| 9 | `narrow_copy` |
| 2 | `narrow_copy.out` |

Native aborts in the portable `narrow_copy` kernel — not OpenVINO.

## Portable-kernel SKIP (866 jobs, 5 operators) — `error 0x12`
Ops that fell to portable kernels which then raised a `Check failed` / unsupported guard
(`_fft_r2c onesided=False`, `index requires int64`, `_pdist_forward` unhandled dtype, distance-util
rank checks, `sub` alpha extraction). Enumerated verbatim in [../skips.md](../skips.md) SKIP class B.
These are portable coverage gaps, not OpenVINO.
