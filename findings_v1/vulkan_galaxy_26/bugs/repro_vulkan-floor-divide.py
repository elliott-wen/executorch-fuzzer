"""Repro: vulkan integer floor_divide truncates toward zero instead of flooring.

Vulkan floor_divide kernel uses the GLSL operator literal `floor(X / Y)`
(pytorch_ref/executorch/backends/vulkan/runtime/graph/ops/glsl/binary_op_buffer.yaml:29-30,
applied at binary_op_buffer.glsl:71/:86; registered in
runtime/graph/ops/impl/BinaryOp.cpp:152 -> :134, dtype-specialized at BinaryOp.cpp:80).

For INTEGER dtype the shader's `X / Y` is GLSL integer division (truncates toward zero)
and the outer `floor()` is a no-op on an integral value -> result is trunc, not floor.
Eager aten.floor_divide on ints floors toward -inf.

Run: .venv/bin/python findings/vulkan_galaxy_26/bugs/repro_vulkan-floor-divide.py
(Eager runs on host; the vulkan column is reproduced arithmetically from the GLSL formula,
matching the on-device localizer values. Device exec not required.)
"""
import math
import torch


def glsl_floor_divide_int(x: int, y: int) -> int:
    """What the vulkan shader computes for integer DTYPE: floor(X / Y) where X/Y is
    C/GLSL integer division (trunc toward zero), then floor() is a no-op on the int."""
    q = int(x / y) if y != 0 else 0          # C-style trunc toward zero
    q = int(q)                                # GLSL int(); floor(int) is identity
    return q


def glsl_floor_divide_fp(x: float, y: float, dtype=torch.float16) -> float:
    """Float DTYPE path: floor(X / Y) but X/Y is first rounded to fp16."""
    q = (torch.tensor(x, dtype=dtype) / torch.tensor(y, dtype=dtype)).item()
    return math.floor(q)


cases_int = [
    ("w56:1050", -4, 9),    # device e=-1 b=0
    ("w16-int",  -15, 2),   # generic: floor=-8, trunc=-7
    ("w34:1524", -1, 4),    # device e=-1 b=0
]
print("== integer floor_divide (the ~81 cluster) ==")
print(f"{'job':<10} {'x//y':<8} {'eager(floor)':<14} {'vulkan floor(X/Y)=trunc':<24}")
for name, x, y in cases_int:
    eager = torch.floor_divide(torch.tensor(x), torch.tensor(y)).item()
    vk = glsl_floor_divide_int(x, y)
    flag = "  <-- DIVERGES" if eager != vk else ""
    print(f"{name:<10} {f'{x}//{y}':<8} {eager:<14} {vk:<24}{flag}")

print("\n== float16 secondary case w16:1546 ==")
x, y = -2.671875, 0.381591796875
eager = torch.floor_divide(torch.tensor(x, dtype=torch.float16),
                           torch.tensor(y, dtype=torch.float16)).item()
vk = glsl_floor_divide_fp(x, y)
print(f"eager={eager}  vulkan floor(fp16(X/Y))={vk}  (device b[0]=-7)")
print("fp16 rounds X/Y to exactly -7.0 so floor(-7.0)=-7; true ratio is -7.0019 -> eager -8")
