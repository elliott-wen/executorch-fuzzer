# `_native_batch_norm_legit.no_stats` — numerical divergence vs ATen

- **Mode:** MISMATCH (finite values) · **Occurrences:** 17 · reason: `max|delta|≈4.0`

The portable batch-norm (no-stats variant) diverges numerically from ATen beyond tolerance on finite
inputs (`(x-μ)/√(σ²+ε)` computed with a different ε placement / reduction order at small batch). Lower
confidence than the non-finite bugs (near-tolerance on some samples — gate with N≥5 before filing as
a hard bug); recorded here as a numerical divergence.
