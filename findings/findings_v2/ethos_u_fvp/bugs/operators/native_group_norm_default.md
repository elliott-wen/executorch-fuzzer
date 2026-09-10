# Ethos-U operator bug — `native_group_norm.default`

> ⚠️ **DROPPED — NOT A CONFIRMED BUG.** Re-verification: **0/5** (never reproduced). The single first-pass GENUINE verdict was near-tolerance noise. See `../../REVERIFICATION.md`.


**Failure mode:** mismatch · **Category:** non-finite (nan/inf) · **Verdict:** NONFINITE_OUT (single-op isolation)
**Device:** Arm Corstone-300 FVP (Ethos-U55, int8/Vela) — device-verified against the PT2E-quantized CPU reference

## What it is
Rebuilt as a **one-op graph fed the op's exact runtime (quantized) inputs**, lowered to Ethos-U, run on the FVP, and diffed against the PT2E CPU reference. The op **diverges as the sole output with finite inputs** → genuine kernel/compiler bug (not an upstream or graph-context artifact).

## Device-verified evidence
- **Divergence:** out[0] non-finite mismatch (nan/inf positions differ)
- **Input dtype(s):** torch.float32
- **eager (reference):** [NaN, NaN, NaN]
- **device (Ethos-U):** [NaN, 0.0, 0.0]
- **worst element:** eager `NaN` vs device `NaN`
- **Source rep:** `w28:227` out[4]

## Reproduce
```
source tmp/ethos_env.sh   # FVP + toolchain on PATH; broker on 127.0.0.1:15574
BISECT_BACKEND=ethos-u BISECT_CORPUS=corpus_v2/ethos-u BISECT_QUANTIZE=1 BROKER_PORT=15574 \
  ISO_DEV_TIMEOUT=200 python tmp/isolate_op.py w28:227 4
```
Prints a JSON verdict; a stable `GENUINE`/`NONFINITE_OUT` with the divergence above confirms the bug.
