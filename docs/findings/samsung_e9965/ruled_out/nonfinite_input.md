# Ruled out — non-finite INPUT propagation (filter 3e)

Several ops MISMATCH as `nan → inf` (or `nan → finite`). The corpus deliberately injects
`nan/inf/-inf/0/-0` **input leaves** ("exercise non-finite / boundary INPUT domains"). Where the
eager reference is *itself* non-finite because the input was non-finite, a differing device
non-finite value is **input-domain handling difference**, not a clean kernel bug (§3e). Verified by
reading each graph's `LEAVES`.

Ruled out (leaf non-finite, eager non-finite):
`gelu.out`, `relu`, `var.correction_out`, `logit`/`logit.out`, `linear`/`linear.out`,
`_softmax.out` (nan-input → device 0), `sqrt.out` / `rsqrt.out` (negative-input → eager nan),
`clamp.out` (inf-input → device saturates to fp16 max 65504, arguably *more* correct).

**NOT ruled out — kept as a confirmed bug:** `div.Scalar` overflows fp16 from a **finite** input
(`[-0.48,1.11,0.93] → [-31600, inf, 61184]`) — a genuine op-generated non-finite. See
`../bugs/repro_div_scalar_overflow.py`.
