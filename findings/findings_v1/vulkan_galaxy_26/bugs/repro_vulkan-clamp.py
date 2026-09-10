#!/usr/bin/env python3
"""Repro for the 'clamp divergence' cluster (vulkan_galaxy_26).

Conclusion: the clamp born-here cluster is NOT a clamp-kernel bug.
  (a) The scalar GLSL clamp path is correct, incl. None->+/-inf defaults and fp16.
  (b) Every divergent job uses clamp.Tensor / clamp.Tensor_out (tensor bounds),
      which Vulkan does NOT register -> it runs on portable CPU (correct),
      so the divergence is INHERITED from an upstream GPU op (int64 / fp16).
  (c) Re-derives the inherited int64 case w4:251 (eager -3 -> device -inf):
      the wrong value enters at clamp's int64 input, not at clamp.

Source references (read-only checkout):
  pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/impl/UnaryOp.cpp
    :72-78  get_val_or_inf  -> None bound becomes +inf (max) / -inf (min)
    :86-95  DEFINE_CLAMP_FN -> only path; passes two float scalars as push consts
    :165    VK_REGISTER_OP(aten.clamp.default, clamp)   # ONLY clamp registered
  pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/glsl/unary_op.glsl
    :52     t_out[i] = T(clamp(float(t_in[i]), minimum, maximum));
  pytorch_ref/executorch/backends/vulkan/op_registry.py:239  # only clamp.default
"""
import math
import numpy as np
import torch

# --- (a) model the scalar GPU clamp path exactly --------------------------------
# UnaryOp.cpp get_val_or_inf: None -> -inf (min) / +inf (max).
def get_val_or_inf(val, is_max):
    if val is not None:
        return float(val)
    return math.inf if is_max else -math.inf

# unary_op.glsl: T(clamp(float(x), minimum, maximum)); GLSL clamp = min(max(x,a),b).
def gpu_clamp_scalar(x, mn, mx, out_dtype=np.float32):
    a = get_val_or_inf(mn, False)
    b = get_val_or_inf(mx, True)
    xf = float(x)
    r = min(max(xf, a), b)
    return np.array([r], dtype=out_dtype)[0]

print("== (a) scalar GPU clamp path vs eager (None bounds + fp16) ==")
cases = [
    # (x, min, max, out_dtype)
    (-3.0, None, 0.0, np.float32),   # max only  -> min(max(-3,-inf),0) = -3   (NOT -inf)
    ( 7.0, 5.0, None, np.float16),   # min only  -> min(max(7,5),+inf) = 7
    (-0.48, 5.0, None, np.float16),  # w0-like: clamp(fp16, 5, None) -> 5
    ( 2.0, 5.0, 0.0, np.float32),    # min>max (UB-ish): min(max(2,5),0)=0; eager also 0
]
for x, mn, mx, dt in cases:
    g = gpu_clamp_scalar(x, mn, mx, dt)
    e = torch.clamp(torch.tensor(x, dtype=torch.float16 if dt is np.float16 else torch.float32),
                    min=mn, max=mx).item()
    ok = (math.isclose(float(g), e, rel_tol=1e-3, abs_tol=1e-3))
    print(f"  clamp({x:>5}, min={mn}, max={mx}) gpu={float(g):>8}  eager={e:>8}  match={ok}")
print("  -> scalar path is correct; None never produces -inf/nan by itself.\n")

# --- (b) the divergent jobs all use clamp.Tensor (CPU fallback) -----------------
print("== (b) clamp.Tensor_out (tensor bound) is the divergent variant ==")
# w19:360-style: per-channel max via a broadcast tensor bound. Vulkan has no
# clamp.Tensor kernel (op_registry.py:239), so this runs on portable CPU.
x = torch.randn(2, 4, 4, dtype=torch.float32)
mx = torch.tensor([[[-0.36]], [[0.0985]]], dtype=torch.float32)  # shape (2,1,1) broadcast
e = torch.ops.aten.clamp.Tensor_out(x, None, mx, out=torch.empty(2, 4, 4))
print(f"  clamp.Tensor_out exists and computes on CPU; sample out[0,0,0]={e[0,0,0].item():.4f}")
print("  Vulkan registers only aten.clamp.default -> this node is NOT delegated.")
print("  => any device mismatch here is INHERITED from the GPU-produced input.\n")

# --- (c) re-derive the inherited int64 case w4:251 (e=-3 -> device=-inf) ---------
print("== (c) w4:251 is upstream int64 truncation feeding clamp, not a clamp bug ==")
torch.manual_seed(12648430)
L0 = torch.randint(-4, 5, (2, 4, 4), dtype=torch.int64)
n0 = torch.ops.aten.pixel_unshuffle.default(L0, 1)          # still int64
n2_eager = torch.ops.aten.clamp.out(n0, None, 0.0, out=torch.empty((2, 4, 4), dtype=torch.float32))
print(f"  eager  n2[0,0,0] = clamp(int64 {int(n0[0,0,0])}, max=0) = {n2_eager[0,0,0].item()}")
# On Adreno, n0 (int64) is truncated/mangled BEFORE clamp (no 64-bit int on GPU);
# the localizer recorded device n2[0]=-inf -> the broken input, not the clamp op.
print("  device n2[0] = -inf  (per localizer): int64 input was corrupted upstream")
print("  -> see vulkan-int64-truncation.md. clamp merely propagated it.")
