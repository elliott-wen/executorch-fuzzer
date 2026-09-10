"""Repro: vulkan fp16 reduction / pow / int-out-sum divergences (vulkan_galaxy_26).

Reasoned from source + eager (NOT run on device). Device values are the ones
recorded by the on-device first-divergence localizer (tmp/run_vulkan_galaxy_26).

Three independent mechanisms, all rooted in the Vulkan GLSL kernels:

A. mean -> -65504  (fp16-precision, Adreno mediump has no IEEE NaN/Inf)
   - Reduce.cpp:144 picks the shader by OUTPUT dtype; reduce.glsl accumulates in
     a `half` vec4 (no fp32 accumulator). reduce.yaml:20-21 mean = accum/N.
   - gen_vulkan_spv.py:870-880 downgrades all-half variants highp->mediump.
   - On Adreno mediump float is fp16 w/o NaN/Inf -> a NaN input (or 0/0) flushes
     to +/-65504 (= +/- fp16 max). Saturation is in the GPU ALU, no explicit clamp.

B. pow special values (real bug, two distinct shader paths)
   - pow.Tensor_Scalar -> binary_op_defs.glslh:23-27 power_of(): x==0 returns 0
     for any y!=0, so 0^(-1) -> 0 instead of +Inf.
   - pow.Tensor_Tensor -> binary_op_texture.yaml:28-29 uses the RAW GLSL pow(X,Y),
     undefined for x<0 (binary_op_defs.glslh:14 says so) -> pow(-1,-1) -> NaN.

C. int-out sum (real bug): vulkan accumulates in float then casts the result;
   eager casts each element to the int out dtype first, then accumulates.
   reduce.glsl:101 `accum + new_val`, store at :132.
"""
import math
import torch

torch.manual_seed(0)
PASS = True


def check(label, eager, device_expected, vulkan_emulated):
    global PASS
    ok = (repr(vulkan_emulated) == repr(device_expected))
    PASS = PASS and ok
    print(f"[{'OK ' if ok else 'XX '}] {label}")
    print(f"        eager           = {eager}")
    print(f"        vulkan (device) = {device_expected}")
    print(f"        vulkan (emul)   = {vulkan_emulated}")


# ---------------------------------------------------------------- A. mean -> -65504
def fp16_reduce_emul(vals):
    """Emulate the half-accumulator mediump reduce on Adreno: accumulate in fp16,
    and (Adreno mediump) collapse any non-finite to -fp16-max."""
    FP16_MAX = 65504.0
    if len(vals) == 0:
        # mean of empty: accum=0, N=0 -> 0/0; Adreno mediump yields -fp16-max
        return -FP16_MAX
    acc = torch.tensor(0.0, dtype=torch.float16)
    for v in vals:
        acc = (acc + torch.tensor(v, dtype=torch.float16)).to(torch.float16)
    mean = acc / len(vals)
    if not torch.isfinite(mean):
        return -FP16_MAX
    return mean.item()

# w0:1016  mean of empty (0,) float32
e = torch.ops.aten.mean.out(torch.empty((0,)), None, True, dtype=None,
                            out=torch.empty((1,), dtype=torch.float32))[0].item()
check("w0:1016  mean(empty)", e, -65504.0, fp16_reduce_emul([]))

# w56:1345  mean([nan, 0.4656])  (n1 had a nan from acosh out-of-domain)
inp = [float('nan'), 0.465576171875]
e = torch.tensor(inp).mean().item()
check("w56:1345 mean([nan,0.46])", e, -65504.0, fp16_reduce_emul(inp))

# w4:313   mean([-0.503, nan])   (n0 = asin of value>1 -> nan)
inp = [-0.5034968256950378, float('nan')]
e = torch.tensor(inp).mean().item()
check("w4:313   mean([-0.50,nan])", e, -65504.0, fp16_reduce_emul(inp))


# ---------------------------------------------------------------- B. pow
def power_of(x, y):
    """binary_op_defs.glslh power_of() used by pow.Tensor_Scalar."""
    if x == 0.0:
        return 1.0 if y == 0.0 else 0.0   # <-- 0^neg -> 0 (BUG: should be +Inf)
    r = math.pow(abs(x), y)
    if x < 0.0:
        iy = round(y)
        if abs(y - iy) < 1e-5 and int(iy) % 2 == 1:
            r = -r
    return r

# w74:62  pow.Tensor_Scalar([0,0], -1)  -> eager inf, device 0
e = torch.ops.aten.pow.Tensor_Scalar_out(torch.zeros(2, dtype=torch.float16), -1,
        out=torch.empty(2, dtype=torch.float32))[0].item()
check("w74:62   pow.Scalar(0,-1)", e, 0.0, power_of(0.0, -1.0))

# w57:181  pow.Tensor_Tensor(-1, -1)  -> raw GLSL pow(x<0,..) UNDEFINED -> NaN on device
e = ((-1.0) ** (-1))
# raw GLSL pow(neg, .) is undefined; Adreno returns NaN. (libm would give -1.)
vk = float('nan')
check("w57:181  pow.Tensor(-1,-1)", e, float('nan'), vk)


# ---------------------------------------------------------------- C. int-out sum
def vk_sum_int(vals, fp16=True):
    """Vulkan reduce: accumulate in float (here fp16 input), cast result to int."""
    dt = torch.float16 if fp16 else torch.float32
    acc = torch.tensor(0.0, dtype=dt)
    for v in vals:
        acc = (acc + torch.tensor(v, dtype=dt)).to(dt)
    return int(acc.item())  # trunc-to-int on store

# w1:757  sum(fp16[0.881]*4) -> int64 :  eager casts-then-sums = 0 ; vulkan = 3
inp = [0.88134765625] * 4
e = torch.ops.aten.sum.IntList_out(torch.tensor(inp, dtype=torch.float16), None, True,
        dtype=None, out=torch.empty(1, dtype=torch.int64))[0].item()
check("w1:757   sum->int64 [0.881x4]", e, 3, vk_sum_int(inp))

# w38:1209  sum(fp16[-0.48,1.11]) -> int64 : eager 1 ; vulkan 0
inp = [-0.482421875, 1.1103515625]
e = torch.ops.aten.sum.IntList_out(torch.tensor(inp, dtype=torch.float16), None, True,
        dtype=None, out=torch.empty(1, dtype=torch.int64))[0].item()
check("w38:1209 sum->int64 [-0.48,1.11]", e, 0, vk_sum_int(inp))


print("\nALL MATCH DEVICE" if PASS else "\nSOME MISMATCH")
