# SKIP reason: type-promotion guards

**Count: 2,140 (5.7% of skips).** The kernel enforces a **stricter type-promotion contract
than eager** and refuses rather than promoting. Where eager would promote the operands and
compute, portable returns an error.

## Clusters

| count | operator | firing file | guard |
|---:|---|---|---|
| 930 | `add.Scalar_out` | `op_add.cpp` | `common_type == a_type && check_alpha_type(...)` |
| 599 | `_native_batch_norm_legit_no_training.out` | `op_native_batch_norm.cpp` | `a.scalar_type() == b.scalar_type()` (e.g. `{Half, Float}`) |
| 214 | `scatter.value_out` | `op_scatter.cpp` | `index.scalar_type()` must be Long |
| 114 | `scatter.src_out` | `op_scatter.cpp` | `index.scalar_type()` must be Long |
| 101 | `scatter_add.out` | `op_scatter_add.cpp` | `index.scalar_type()` must be Long |
| 27 | `remainder.Scalar_out` | `op_remainder.cpp` | `!isIntegralType(common_type)` |
| 26 | `fmod.Scalar_out` | `op_fmod.cpp` | `!isIntegralType(common_type)` |
| 23 | `pow.Tensor_Scalar_out` | `op_pow.cpp` | `canCast(common_type, out.scalar_type())` |

## Notes

- **`add.Scalar` common_type/alpha guard (930).** Portable requires the scalar's promoted type
  to match the input type and a valid alpha type; eager promotes more liberally. The largest
  single promotion-guard cluster.
- **`batch_norm` input vs running-stat dtype must match (599).** Eager promotes mixed
  `{Half, Float}`; portable refuses. (Related dtype-strictness lives in the same kernel as the
  [batch_norm `1/sqrt(0)` bug](../bugs/normalization-bornhere.md).)
- **`scatter*` index must be Long (429).** Standard — eager also requires Long indices.
- **`remainder`/`fmod` reject integral common_type (53)** for the `.Scalar` overload — note the
  *Tensor* overloads do NOT have this guard and instead produce the
  [integer remainder-sign bug](../bugs/remainder-sign.md) (mismatch) or a
  [SIGFPE on a zero divisor](../crash/03-integer-divide-by-zero-sigfpe.md) (crash). So the
  guard is inconsistent across overloads.

## Classification
Mostly working-as-intended (stricter-but-safe). The `add.Scalar` and `batch_norm` dtype
strictness are the cases where portable diverges from eager's promotion — worth noting as a
portability constraint, but they fail gracefully (not bugs). The interesting observation is the
**inconsistency**: the `.Scalar` remainder/fmod overloads guard integral types while the
`.Tensor` overloads don't — and the unguarded ones are exactly where the real crash/mismatch
bugs live.
