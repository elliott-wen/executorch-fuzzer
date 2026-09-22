# Vulkan delegate: bitwise_left_shift over-width shift wraps instead of zeroing

**Signature:** `x << n` with `n >= bit-width` gives `x << (n mod width)` on Vulkan, but `0` in eager.
**Classification:** real-bug — **kernel / shift semantics**, not aliasing.
**Status:** **device-confirmed on the Moto G54 5G** in isolation, 2026-06-29.

## What happens

For an `int32` tensor, a left shift by `40` (≥ the 32-bit width):
- **eager (CPU / PyTorch):** an over-width shift is defined as **0**.
- **Vulkan / Mali GPU:** the GLSL shift operator masks the count to the type width, so it shifts
  by `40 mod 32 = 8`, giving `x * 256`.

Isolated single-op job:

```
x      = int32 [1, 2, 3, 4],  shift = 40
eager  = [0, 0, 0, 0]
device = [256, 512, 768, 1024]    <-- shifted by (40 mod 32)=8 instead of 0
```

Smaller in-range shifts agree on the phone (`int8 << 5`, `uint8 << 1` both OK) — the divergence is
specifically the **over-width** shift count.

`bitwise_right_shift` has the same bug — verified on the phone:

```
int32 [256, 512, 768, 1024] >> 40
eager  = [0, 0, 0, 0]
device = [1, 2, 3, 4]    <-- shifted by (40 mod 32)=8 instead of 0
```

## Mechanism
The observed device result is `x << (n mod width)` (verified for both shift directions above),
consistent with SPIR-V/GLSL shift operators masking the count to the type width, whereas PyTorch
defines an over-width shift as 0. The exact lowering line is not pinned here; the phone
input→output pairs above are what is asserted.

## Repro
```
python mobile/findings_v2/vulkan_moto/bugs/repro_vulkan-left-shift-overwidth.py
```
Needs a connected Vulkan worker (broker on `127.0.0.1:15554`).

## Notes
- Vulkan-specific (portable/eager correct).
- Single-op isolation rules out aliasing — true kernel/semantics bug.
