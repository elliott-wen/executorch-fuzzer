# Ethos-U operator bug — `bitwise_left_shift.Tensor_out`

> ⚠️ **INTERMITTENT — 3/5 reproductions.** Non-deterministic (wrong on some runs, correct on others) — report as intermittent, not a stable bug. See `../../REVERIFICATION.md`.


**Failure mode:** mismatch · **Category:** dtype divergence · **Verdict:** GENUINE (single-op isolation)
**Device:** Arm Corstone-300 FVP (Ethos-U55, int8/Vela) — device-verified against the PT2E-quantized CPU reference

## What it is
Rebuilt as a **one-op graph fed the op's exact runtime (quantized) inputs**, lowered to Ethos-U, run on the FVP, and diffed against the PT2E CPU reference. The op **diverges as the sole output with finite inputs** → genuine kernel/compiler bug (not an upstream or graph-context artifact).

## Device-verified evidence
- **Divergence:** out[0] dtype torch.float32 vs torch.uint8
- **Input dtype(s):** torch.int32, torch.int8
- **eager (reference):** [0.0, 0.0, 0.0, 0.0, -4.0]
- **device (Ethos-U):** [0.0, 0.0, 1.0, 0.0, 0.0]
- **worst element:** eager `null` vs device `null`
- **Source rep:** `w13:656` out[2]

## Reproduce
```
source tmp/ethos_env.sh   # FVP + toolchain on PATH; broker on 127.0.0.1:15574
BISECT_BACKEND=ethos-u BISECT_CORPUS=corpus_v2/ethos-u BISECT_QUANTIZE=1 BROKER_PORT=15574 \
  ISO_DEV_TIMEOUT=200 python tmp/isolate_op.py w13:656 2
```
Prints a JSON verdict; a stable `GENUINE`/`NONFINITE_OUT` with the divergence above confirms the bug.
