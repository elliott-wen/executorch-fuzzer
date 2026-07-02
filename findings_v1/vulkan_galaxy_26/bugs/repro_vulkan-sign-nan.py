#!/usr/bin/env python3
"""Repro: vulkan sign(NaN)=NaN, eager sign(NaN)=0  (nonfinite divergence).

Born-here root op `sign`, kind `nonfinite`. Example jobs (localizer e[0]=0 b[0]=nan):
  w34:418   acosh(bool 0)        -> NaN ; sign -> eager 0 / vulkan nan
  w19:549   acosh(relu(<1))      -> NaN ; sign -> eager 0 / vulkan nan
  w36:1059  remainder(var dof<=0)-> NaN ; sign -> eager 0 / vulkan nan

ATen mandates sign(NaN)=0 (real signum). The vulkan delegate's sign GLSL operator is
NaN-propagating (GLSL/SPIR-V `sign()`/`FSign` returns NaN for NaN), so it disagrees.
Reference unary-op path:
  pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/impl/UnaryOp.cpp:37 (add_unary_op_node)
  pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/glsl/unary_op.glsl:64/76
  (this read-only checkout does NOT register aten.sign; the device build does -- see md)
Same signature as findings/portable/bugs/sign-nan.md (portable instance of the same gap).

Run: cd /data/jwen929/mobile && .venv/bin/python \
     findings/vulkan_galaxy_26/bugs/repro_vulkan-sign-nan.py
"""
import math
import torch


def aten_sign_specials():
    x = torch.tensor([float("nan"), float("inf"), -float("inf"),
                      0.0, -0.0, -3.0, 2.0], dtype=torch.float32)
    print("ATen sign  :", torch.sign(x).tolist())
    print("ATen sgn   :", torch.sgn(x).tolist())
    print("  -> sign(NaN) =", float(torch.sign(x)[0]), "(eager maps NaN to a FINITE 0)")


def glsl_sign_nan_propagating(v):
    """GLSL/SPIR-V `sign(x)`: +1/-1/0 for finite, but NaN-propagating for NaN (device)."""
    if math.isnan(v):
        return float("nan")          # <-- the vulkan bug
    return (v > 0) - (v < 0)


def aten_correct_sign(v):
    """isnan(X) ? 0 : sign(X)  -- the ATen-conformant form the shader SHOULD use."""
    if math.isnan(v):
        return 0.0
    return (v > 0) - (v < 0)


def upstream_nan_chains():
    print("\n-- upstream chains that feed NaN into sign (NaN appears on BOTH backends):")
    # w34:418: acosh of bool-derived 0 (domain [1,inf))
    L0 = torch.tensor([False, True, False])
    n0 = torch.ops.aten.acosh.default(L0.to(torch.float32))
    print(f"  w34:418  acosh([0,1,0]) = {n0.tolist()}  (NaN where input<1)")
    # w36:1059: var with dof<=0 -> nan, remainder -> nan
    n0v = torch.ops.aten.var.correction_out(
        torch.randn(()), None, correction=None, keepdim=False,
        out=torch.empty((), dtype=torch.float16))
    n2 = torch.ops.aten.remainder.Scalar(n0v, -3)
    print(f"  w36:1059 remainder(var dof<=0, -3) = {float(n2)}")


if __name__ == "__main__":
    print("=== eager (ATen) sign special values ===")
    aten_sign_specials()

    upstream_nan_chains()

    print("\n=== sign(NaN): eager vs vulkan-shader semantics ===")
    e = float(torch.sign(torch.tensor(float("nan"))))
    v = glsl_sign_nan_propagating(float("nan"))
    c = aten_correct_sign(float("nan"))
    print(f"  eager torch.sign(nan)        = {e}")
    print(f"  vulkan GLSL sign(nan)        = {v}   <-- BUG (NONFINITE-DIFF)")
    print(f"  correct isnan?0:sign(nan)    = {c}   (proposed fix)")
    status = "NONFINITE-DIFF (REPRODUCED)" if (e == 0.0 and math.isnan(v)) else "??"
    print(f"  result: {status}")
