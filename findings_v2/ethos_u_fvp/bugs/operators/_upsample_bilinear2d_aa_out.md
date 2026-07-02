# Ethos-U operator bug — `_upsample_bilinear2d_aa.out`

> ⚠️ **INTERMITTENT — 1/5 reproductions.** Non-deterministic (wrong on some runs, correct on others) — report as intermittent, not a stable bug. See `../../REVERIFICATION.md`.


**Failure mode:** mismatch · **Category:** wrong value · **Verdict:** GENUINE (single-op isolation)
**Device:** Arm Corstone-300 FVP (Ethos-U55, int8/Vela) — device-verified against the PT2E-quantized CPU reference

## What it is
Rebuilt as a **one-op graph fed the op's exact runtime (quantized) inputs**, lowered to Ethos-U, run on the FVP, and diffed against the PT2E CPU reference. The op **diverges as the sole output with finite inputs** → genuine kernel/compiler bug (not an upstream or graph-context artifact).

## Device-verified evidence
- **Divergence:** out[0] max|delta|=2.000e+00 (rtol=0.0 atol=0.0)
- **Input dtype(s):** torch.uint8
- **eager (reference):** [6.0, 6.0, 6.0, 6.0, 248.0, 3.0]
- **device (Ethos-U):** [6.0, 6.0, 5.0, 5.0, 248.0, 1.0]
- **worst element:** eager `3.0` vs device `1.0`
- **Source rep:** `w16:743` out[1]

## Reproduce
```
source tmp/ethos_env.sh   # FVP + toolchain on PATH; broker on 127.0.0.1:15574
BISECT_BACKEND=ethos-u BISECT_CORPUS=corpus_v2/ethos-u BISECT_QUANTIZE=1 BROKER_PORT=15574 \
  ISO_DEV_TIMEOUT=200 python tmp/isolate_op.py w16:743 1
```
Prints a JSON verdict; a stable `GENUINE`/`NONFINITE_OUT` with the divergence above confirms the bug.
