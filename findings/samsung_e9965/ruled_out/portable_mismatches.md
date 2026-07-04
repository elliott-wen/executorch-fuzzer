# Ruled out — portable-fallback mismatches (filter 3a)

**1,299** MISMATCH/CRASH outcomes ran on the portable CPU kernel (`ops=0`), i.e. the op was NOT
delegated to ENN for that graph. A divergence there is a portable-kernel or reference issue, not a
bug in the Samsung/ENN backend under test. Excluded from all ENN findings. Notable ops that are
mostly-portable (so their raw mismatch counts are misleading): `bitwise_left_shift.Tensor_out`
(137 portable vs 58 delegated), `remainder.Tensor_out` (62 vs 113). Only the delegated subset is
filed against ENN.
