# CoreML `mean` over integer input — integer-truncated mean instead of float mean

- **Failure mode:** MISMATCH
- **Root cause:** operator kernel (delegated to Core ML; `delegated.ops=1`, `non_delegated=0`)
- **Mechanism:** WRONG-VALUE — the mean is computed in integer arithmetic (truncated toward 0)
  instead of promoting to float
- **Occurrences:** 69 delegated mismatches (`mean`). Device-verified, deterministic (5/5).
  (Note: `mean.out` also has a native-abort **crash** form — see [crash-native-abort.md](crash-native-abort.md).)
- **Repro:** [repro_mean-int.py](repro_mean-int.py) (corpus jobs `w100:145`, `w101:549`)

## What happens
`torch.mean` of an integer tensor promotes to floating point and returns the true average. The Core ML
delegate returns a value consistent with an **integer** mean (sum then integer-divide, truncating
toward zero):

```
job w100:145   mean(L0: int32)   ->  float32 out
  eager : [-0.708333]      # true float mean
  device: [0.0]            # trunc(-0.708) = 0
job w101:549
  eager : [-1.5]
  device: [-1.0]           # trunc(-1.5) = -1
```

Both `-0.708→0` and `-1.5→-1` are exactly truncation-toward-zero of the correct mean → the division
is done in integer space before (or instead of) the float promotion.

## Why
For an integer input, `mean` must upcast to float *before* dividing by the element count. The lowering
appears to divide in the input (integer) dtype, truncating the quotient, then cast the truncated
integer to the float output. Ordinary in-distribution integer leaves trigger it.
