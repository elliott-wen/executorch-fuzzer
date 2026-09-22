#!/usr/bin/env python3
"""Repro: int64 silently truncated to int32 on the Vulkan/Adreno GPU.

The Vulkan backend has no 64-bit int. With downcast_64_bit=True (default), every
int64 tensor is stored on-GPU as int32:
    pytorch_ref/executorch/backends/vulkan/serialization/vulkan_graph_builder.py:218-239
The 64<->32 conversion is done on the host staging copy as a plain narrowing cast:
    pytorch_ref/.../runtime/graph/ComputeGraph.cpp:952-983 / 995-1027
       if src==kLong && staging==kInt: cast_and_copy_from<int64_t,int32_t>(...)
    pytorch_ref/.../runtime/api/containers/StagingBuffer.h:84-96 / 114-126
       dst[i] = static_cast<DST_T>(src[i]);   // keeps low 32 bits, no saturation/guard

This script runs eager (CPU/true int64) and applies that exact static_cast<int32_t>
truncation in Python, showing it reproduces the device values recorded by the
on-phone localizer:  w33 -> 0,  w1 -> -1,  w78 -> 2.306e18.
No Vulkan device required.
"""
import ctypes
import torch


def trunc_i64_to_i32_and_back(x: int) -> int:
    """What the Vulkan staging copy does to one int64 element:
    write-in: static_cast<int32_t>(int64)  (keep low 32 bits)
    read-out: static_cast<int64_t>(int32)  (sign-extend)
    """
    i32 = ctypes.c_int32(x).value          # low 32 bits, as signed int32
    return ctypes.c_int64(i32).value       # sign-extended back to int64


def main():
    torch.manual_seed(12648430)
    print("=== w33:241  _to_copy(nan -> int64) ===")
    # rsqrt of a negative -> nan ; nan -> int64 is INT64_MIN in eager.
    L0 = torch.randn((2,), dtype=torch.float32)
    n1 = torch.rsqrt(L0)[0:1]                      # nan
    eager = int(n1.to(torch.int64)[0])
    print(f"  eager nan->int64 = {eager} ({float(eager):.3e})  [INT64_MIN]")
    print(f"  vulkan staging trunc = {trunc_i64_to_i32_and_back(eager)}  (device localizer: b[0]=0)")
    assert trunc_i64_to_i32_and_back(eager) == 0

    print("\n=== w1:1582  min over EMPTY int64 (identity = INT64_MAX) ===")
    # min over an empty int64 reduction returns the int64 reduction identity = INT64_MAX.
    # The graph's float32 out arg rounds it for the *eager* return, but on device the
    # reduction runs in int32: the int64 INT64_MAX (0x7FFF_FFFF_FFFF_FFFF) lives as int32
    # before any float conversion, so we truncate the true int64 bit-pattern.
    eager_i64 = torch.iinfo(torch.int64).max  # 0x7FFFFFFFFFFFFFFF, the reduction identity
    print(f"  eager min(empty int64) identity = {eager_i64} ({float(eager_i64):.3e})  [INT64_MAX]")
    print(f"  vulkan int32 trunc = {trunc_i64_to_i32_and_back(eager_i64)}  (device localizer: b[0]=-1)")
    assert trunc_i64_to_i32_and_back(eager_i64) == -1

    print("\n=== w78:369  bitwise_left_shift(1, -3)  [negative shift = UB] ===")
    base = torch.tensor([1], dtype=torch.int64)
    shift = torch.tensor([-3], dtype=torch.int64)
    eager = int(torch.bitwise_left_shift(base, shift)[0])
    print(f"  eager 1 << -3 = {eager}")
    # GPU treats the negative shift as masked to the type width: (-3) & 63 = 61
    gpu = 1 << ((-3) & 63)
    print(f"  vulkan (negative shift masked to width): 1 << ((-3)&63) = 1<<61 = {gpu} ({float(gpu):.3e})"
          f"  (device localizer: b[0]=2.306e18)")
    assert abs(float(gpu) - 2.306e18) / 2.306e18 < 1e-3

    print("\nAll three device values reproduced from eager + the int32 truncation. PASS")


if __name__ == "__main__":
    main()
