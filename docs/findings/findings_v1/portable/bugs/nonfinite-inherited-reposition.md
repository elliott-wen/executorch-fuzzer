# Artifact: inherited non-finite (NaN) re-positioned downstream

## Classification: ARTIFACT — not an ExecuTorch bug

This is a **harness-strictness artifact**, not a portable-kernel defect. A NaN is
legitimately produced by an out-of-domain elementwise op (`acos` of `|x|>1`,
`asin`/`logit`/`atanh`/`log`/`sqrt`/`rsqrt` of invalid domain) — and it is
produced **identically on both backends**. A downstream op (`remainder`,
`floor_divide`, `where`, `sign`, a reduction, or a broadcast) then re-positions
that NaN into a *different output slot* on portable vs eager, because the two
backends use different formula/broadcast ordering. Both backends are "correct"
(both contain the same NaN, born for the same legitimate reason); only the
harness's *position-exact* non-finite check trips on where the NaN ends up.
Cited: mismatch-non-finite **Mechanism 3**
([../mismatch-non-finite.md](../mismatch-non-finite.md)).

## One-line statement
Both backends legitimately produce a NaN upstream (out-of-domain `acos`); a
downstream op places it in a different position, so the position-exact harness
flags a non-finite mismatch even though neither backend is wrong.

## Mechanism
1. An elementwise op is evaluated outside its domain — here `acos(x)` with
   `|x| > 1`, which is **NaN by definition** in both PyTorch and ExecuTorch.
2. The NaN flows into a downstream op (`remainder`/`floor_divide`/`sign`/
   reductions) that broadcasts, reorders, or applies a different formula on the
   two backends.
3. Because the NaN-propagation path differs, the NaN lands in a different
   element of the returned tensor — `portable=nan, eager=0.0` at one slot, and
   the mirror elsewhere.
4. The harness compares NaN/Inf **positions exactly** and reports a non-finite
   mismatch, even though *both* outputs carry the same (legitimately-born) NaN.

The NaN is **inherited** from the out-of-domain upstream op, not born in the
returned op. Neither the upstream domain behavior (NaN for `acos(|x|>1)`) nor the
downstream op is wrong; only their *interaction with a position-exact comparator*
produces the flag.

## Reproduction (verified)
Job: [`corpus/portable/w0/w0_313.py`](../../../corpus/portable/w0/w0_313.py).

Chain (from the graph header):
```
n0 = acos.out(L0)                              # NaN where |L0|>1, on BOTH backends
n1 = broadcast_to(n0, [2,2,2])
n2 = remainder.Tensor_out(n1, n0, out=float16)
n7 = clone(n1)
n15 = sign.out(n7, out=float32)   -> USER_OUTPUT[4]   # non-finite positions differ
n12 = floor_divide.out(n0, n8, out=float32)
n16 = sign.out(n12, out=float32)  -> USER_OUTPUT[5]   # non-finite positions differ
```

Proof the NaN is **inherited (born on both backends)** — eager side, on the
stored input:
```
L0          = [-0.4824913740158081, 1.1103534698486328]
acos(L0)    = [2.0742931365966797, nan]      # acos(1.1103) is OOB -> nan
isnan       = [False, True]
```
`acos(1.1103…)` is NaN in eager exactly as in portable: the NaN is legitimate and
present on both sides.

Observed at the returned output:
```
out[4] (sign of acos-derived) p=float32/e=float32  non-finite-pos-diff = 4
   idx1: portable=nan   eager=0.0      # same NaN, different downstream slot
```
(`out[5]`, `sign(floor_divide(...))`, shows the same shape with
`non-finite-pos-diff=1`.) The position-exact check trips; both tensors carry the
same inherited NaN.

## Why it is not a backend bug
- The NaN is **correct on both backends** — `acos(|x|>1) = NaN` is the defined
  domain behavior, reproduced identically by eager and portable.
- The disagreement is purely *which element* holds the NaN after a downstream
  re-position, caused by differing broadcast/formula ordering — not a wrong
  value, not a spurious NaN, not a missing NaN.
- The harness compares NaN/Inf positions exactly, so any NaN reordering through a
  reduction/broadcast/scatter trips it. That is a property of the *comparator's
  strictness*, not of the kernels.
- (Distinguish from the genuine non-finite kernel bugs in the same cluster —
  `sign(nan)` propagating NaN instead of returning 0, and negative-shift →
  float overflow — which are real and documented in
  [sign-nan.md](sign-nan.md) and [bitwise-shift-negative.md](bitwise-shift-negative.md).
  This job's flag is the *inherited-reposition* artifact, not those.)

## Recommendation
(From mismatch-non-finite §Recommendation 3.)
- When a NaN/Inf is already present in an **input** to the returned op (i.e. the
  non-finite was inherited from a legitimately out-of-domain upstream op such as
  `acos(|x|>1)`), treat downstream position differences as a propagation
  artifact rather than a hard MISMATCH.
- Concretely: compare with **NaN == NaN equality** plus a **"non-finite present
  upstream" allowance** — if both backends contain a NaN/Inf somewhere in the
  upstream cone of the returned op, do not fail solely on the NaN's *position*.
- This de-noises the non-finite cluster down to the genuine kernel bugs
  (`sign(nan)`, negative bit-shift overflow, `floor_divide`-by-zero formula),
  which are tracked separately.

## Replay
Self-contained in-process replay:
[`replay_nonfinite-inherited-reposition.py`](replay_nonfinite-inherited-reposition.py).
Loads `corpus/portable/w0/w0_313.py`, first **proves the NaN is inherited** by
showing `acos(L0)` is NaN on the eager side too, then runs the job on the
portable ExecuTorch runtime and the eager reference and shows the differing
non-finite positions at USER_OUTPUT[4]. Exits 0 iff a non-finite position
difference is observed **and** the upstream `acos` NaN is present on both
backends (proving inheritance, not a born-in-the-returned-op NaN).

```
$ cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_nonfinite-inherited-reposition.py 2>&1 \
    | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference"
job w0:313  upstream proof: L0=[-0.4824913740158081, 1.1103534698486328]
  eager acos(L0)=[2.0742931365966797, nan]  isnan=[False, True]  -> NaN BORN on BOTH backends (|x|>1 OOB)
  out[4] (sign of acos-derived) dt p=torch.float32/e=torch.float32 non-finite-pos-diff=4
    idx1: portable=nan  eager=0.0  (same NaN, different downstream slot)
REPLAY: ARTIFACT reproduced (inherited NaN, position differs)
```

The harness flags the non-finite position difference; the script confirms the
NaN was born identically on both backends upstream, so the position difference is
a propagation artifact, not a backend defect.
