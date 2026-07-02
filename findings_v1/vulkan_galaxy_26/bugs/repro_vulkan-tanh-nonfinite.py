"""Repro: vulkan `tanh` drops NaN / saturates to a finite -1.

Root op: aten.tanh.out  (kind=nonfinite, born-here)
Example job: corpus/vulkan/w0/w0_1253.py, node n5 = tanh.out(n4).

Vulkan source (read-only ref):
  pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/impl/UnaryOp.cpp:151
      DEFINE_ACTIVATION_FN(tanh);                       -> shader name "tanh"
  pytorch_ref/.../graph/ops/glsl/unary_op.yaml:37-38
      - NAME: tanh
        OPERATOR: tanh(clamp(X, -15.0, 15.0))           <-- offending line
  pytorch_ref/.../graph/ops/glsl/unary_op.glsl:51 / :64  (op applied per element)

Mechanism: the clamp(X,-15,15) wrapper is not NaN-safe on the GPU. GLSL/SPIR-V
clamp == min(max(x,lo),hi); min/max with a NaN operand are implementation-defined
and Adreno returns the non-NaN bound. So clamp(nan,-15,15) -> -15, tanh(-15) ~ -1.
A correct tanh(NaN) must stay NaN (IEEE-754 / PyTorch eager).

Device value (recorded by localizer): e[0]=nan  b[0]=-1.
"""
import torch

# Rebuild the tanh input from w0:1253: remainder(x, 0) -> nan, in eager AND on device.
torch.manual_seed(12648430)
L0 = torch.randint(-4, 5, (2,), dtype=torch.int64)
n0 = torch.ops.aten.pow.Tensor_Scalar_out(L0, 2, out=torch.empty((2,), dtype=torch.int64))
n2 = torch.ops.aten.logical_xor.default(n0, n0)              # all zeros
n3 = torch.ops.aten.div.Scalar(n0, -6)
n4 = torch.ops.aten.remainder.Tensor_out(
    n3, n2, out=torch.empty((2,), dtype=torch.float32))      # remainder(x, 0) = nan

print("n4 (tanh input):", n4)                                # tensor([nan, nan])

# Eager (correct): NaN propagates.
eager = torch.ops.aten.tanh.out(n4, out=torch.empty((2,), dtype=torch.float16))
print("eager  tanh(n4):", eager)                             # tensor([nan, nan])
assert torch.isnan(eager).all(), "eager tanh must propagate NaN"

# Host CPU emulation of the GLSL OPERATOR `tanh(clamp(X,-15,15))`: CPU clamp is
# NaN-propagating, so it still gives nan -- the bug is NOT visible on CPU.
host_clamped = torch.tanh(torch.clamp(n4, -15.0, 15.0))
print("host  tanh(clamp(n4,-15,15)):", host_clamped)         # tensor([nan, nan])

# Adreno GPU emulation: GLSL clamp(nan,lo,hi) returns the non-NaN bound (-15),
# so the saturated, finite result the device actually produced is:
def glsl_clamp_nan(x, lo, hi):
    # min(max(x, lo), hi) with Adreno's "NaN -> other operand" min/max
    return torch.where(torch.isnan(x), torch.tensor(lo), torch.clamp(x, lo, hi))

device_like = torch.tanh(glsl_clamp_nan(n4, -15.0, 15.0)).to(torch.float16)
print("device-emulated tanh:", device_like)                  # tensor([-1., -1.])
assert torch.allclose(device_like, torch.full_like(device_like, -1.0)), "expected -1"

print("\nMISMATCH: eager=nan  vs  vulkan(device)=-1  (recorded e[0]=nan b[0]=-1)")
print("Classification: real-bug -- saturating clamp wrapper swallows NaN on GPU.")
