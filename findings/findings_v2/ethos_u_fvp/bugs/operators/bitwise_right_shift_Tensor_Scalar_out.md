# Ethos-U operator bug — `bitwise_right_shift.Tensor_Scalar_out`

> ⚠️ **DROPPED — NOT A CONFIRMED BUG.** Re-verification: **0/5** (never reproduced). The single first-pass GENUINE verdict was near-tolerance noise. See `../../REVERIFICATION.md`.


**Failure mode:** mismatch · **Category:** wrong value · **Verdict:** GENUINE (single-op isolation)
**Device:** Arm Corstone-300 FVP (Ethos-U55, int8/Vela) — device-verified against the PT2E-quantized CPU reference

## What it is
Rebuilt as a **one-op graph fed the op's exact runtime (quantized) inputs**, lowered to Ethos-U, run on the FVP, and diffed against the PT2E CPU reference. The op **diverges as the sole output with finite inputs** → genuine kernel/compiler bug (not an upstream or graph-context artifact).

## Device-verified evidence
- **Divergence:** out[0] max|delta|=8.000e+00 (rtol=0.01 atol=0.001)
- **Input dtype(s):** torch.bool
- **eager (reference):** [0.0]
- **device (Ethos-U):** [8.0]
- **worst element:** eager `0.0` vs device `8.0`
- **Source rep:** `w14:1014` out[1]

## Reproduce
```
source tmp/ethos_env.sh   # FVP + toolchain on PATH; broker on 127.0.0.1:15574
BISECT_BACKEND=ethos-u BISECT_CORPUS=corpus_v2/ethos-u BISECT_QUANTIZE=1 BROKER_PORT=15574 \
  ISO_DEV_TIMEOUT=200 python tmp/isolate_op.py w14:1014 1
```
Prints a JSON verdict; a stable `GENUINE`/`NONFINITE_OUT` with the divergence above confirms the bug.
