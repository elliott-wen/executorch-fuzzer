# XNNPACK fp32 activation/elementwise non-finite — SPLIT into three per-bug files

This cluster (XNNPACK fp32/fp16 activation & elementwise ops producing **finite, wrong** results
where eager produces **NaN/Inf**, *only* under the XNNPACK delegate — the portable backend is
correct) has been split into one file per mechanism, each with a co-located buggy model graph:

| mechanism | bug file | graph | born-here example |
|---|---|---|---|
| **(A) min/max/clamp NaN-launder** (`relu`/`clamp`/`hardtanh`/`minimum`/`maximum`) | [xnnpack-clamp-minmax-nan.md](xnnpack-clamp-minmax-nan.md) | [`graph`](graph_xnnpack-clamp-minmax-nan.py) | `hardtanh(nan)=-2` (`w0:49`) |
| **(B) sqrt/rsqrt negative domain** | [xnnpack-sqrt-rsqrt-domain.md](xnnpack-sqrt-rsqrt-domain.md) | [`graph`](graph_xnnpack-sqrt-rsqrt-domain.py) | `sqrt(-3)=-0` (`w0:963`) |
| **(C) exp NaN/overflow** | [xnnpack-exp-nonfinite.md](xnnpack-exp-nonfinite.md) | [`graph`](graph_xnnpack-exp-nonfinite.py) | `exp(nan)=0` (`w0:938`) |

Common theme: XNNPACK's SIMD min/max and fast-approximation microkernels **launder non-finite
values to finite ones** (`fmin`/`fmax` return the non-NaN operand; the rsqrt-based sqrt and the
rational-poly exp have no non-finite guard). All three are real XNNPACK-delegate correctness
divergences and only bite when the delegate is active. See [README.md](README.md) and the full
replay index [../../portable/REPLAY.md](../../portable/REPLAY.md) convention.
