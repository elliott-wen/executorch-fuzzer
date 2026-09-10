#!/usr/bin/env python3
"""Repro: vulkan _log_softmax launders NaN into a finite value; eager propagates NaN.

Born-here root op `_log_softmax`, kind `nonfinite`. Example jobs:
  w14:1512  eager out[0]=nan  vs  vulkan(device)=0
  w22:1096  eager out[0]=nan  vs  vulkan(device)=85.19
  w14:138   eager out[0]=nan  vs  vulkan(device)=nan   (same op, different layout)

Root cause (vulkan GLSL shader, shared softmax.glsl with OPERATOR2 = X - log(Y)):
  pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/impl/Softmax.cpp:124-125
      (log_softmax kernel name) ; softmax.yaml log_softmax_texture3d variant
  softmax.glsl:96-104   explicit nan/inf scrub: zeroes any lane whose exponent is all-ones
  softmax.glsl:94 / :177 , softmax_buffer.glsl:106   denom clamp max(denom, 1e-37)
  softmax.glsl:62/69/134/147 , softmax_buffer.glsl:79/87   max() over NaN is impl-defined
A correct log_softmax propagates NaN across the whole reduced row (eager does).

Run: cd /data/jwen929/mobile && .venv/bin/python \
     findings/vulkan_galaxy_26/bugs/repro_vulkan-logsoftmax-nonfinite.py
"""
import math
import torch


def eager_log_softmax(row, dtype=torch.float16):
    t = torch.tensor(row, dtype=dtype)
    return torch.ops.aten._log_softmax.default(t, 0, False)


def shader_log_softmax(row):
    """Reproduce the vulkan softmax.glsl arithmetic for a 1-D reduction row.

    Mirrors: rowmax via max() that keeps the finite operand over NaN (Adreno FMax),
    denom = sum(exp(x - rowmax)) clamped to >= 1e-37, output = (x - rowmax) - log(denom),
    then the nan/inf scrub (softmax.glsl:96-104) zeroes any non-finite output lane.
    """
    finite = [v for v in row if math.isfinite(v)]
    rowmax = max(finite) if finite else 0.0          # max() drops the NaN lane
    denom = sum(math.exp(v - rowmax) for v in finite)
    denom = max(denom, 1e-37)                          # softmax.glsl:94 / softmax_buffer.glsl:106
    out = []
    for v in row:
        y = (v - rowmax) - math.log(denom)
        if not math.isfinite(y):                       # softmax.glsl:96-104 nan/inf scrub
            y = 0.0
        out.append(y)
    return out


def show(name, row):
    e = eager_log_softmax(row)
    s = shader_log_softmax(row)
    print(f"--- {name}: input row = {row}")
    print(f"  eager  _log_softmax : {[round(float(x), 3) if torch.isfinite(torch.tensor(x)) else float(x) for x in e.tolist()]}")
    print(f"  vulkan (shader sim) : {[round(x, 3) for x in s]}")
    e0 = float(e[0])
    s0 = s[0]
    print(f"  out[0]: eager={e0}  vulkan={s0}  -> "
          f"{'NONFINITE-DIFF (REPRODUCED)' if (math.isnan(e0) and math.isfinite(s0)) else 'match/other'}")
    print()


if __name__ == "__main__":
    # w22:1096: acos(>1)->nan, sub -> [6.22, nan]; eager all-nan, device gave 85.19 (finite)
    show("w22:1096-like [6.22, nan]", [6.22, float("nan")])

    # w14:1512-like: rsqrt of a negative -> nan in one element of the reduced column;
    # the non-packed texture path scrubs the nan output to 0 (device gave 0).
    show("w14:1512-like [nan, 0.98]", [float("nan"), 0.98])

    print("Eager propagates NaN across the whole row; the vulkan shader scrubs nan/inf "
          "(softmax.glsl:96-104) and clamps the denominator (softmax.glsl:94), yielding a "
          "finite value -> nonfinite mismatch.")
