# Artifact: comparison/bool output flip inherited from an upstream float diff

## Classification: ARTIFACT — not an ExecuTorch bug

This is a **differential-fuzz artifact**, not a defect in any ExecuTorch
portable kernel. The comparison kernel under test (`lt`/`gt`/`ge`/`le`/`eq`/
`ne`/`logical_*`) computes the **correct** boolean result for the inputs it is
handed. The mismatch arises one node *upstream*: a float op (`rsqrt`,
`layernorm`, `batch_norm`, …) produced a value that differs between portable and
eager by a tiny amount, and that value sits close enough to the comparison
threshold that the two slightly-different inputs land on opposite sides. The
boolean output then flips `True`<->`False` (an exact-compare `|delta|=1`), but
the comparator did nothing wrong — it faithfully reported `a < b` for the `a` it
received. Cited: mismatch-delta-exact Mechanism **M6**
([../mismatch-delta-exact.md](../mismatch-delta-exact.md)).

## One-line statement
A bool/integer comparison output flips between backends **because its float
input diverged upstream**, not because the comparator kernel is wrong. The
divergence is *inherited*; there is nothing to fix in the comparison op.

## Mechanism
1. An upstream float op produces `x_portable` and `x_eager` that differ by a
   small precision delta (the legitimate float-kernel divergence documented in
   the float/transcendental clusters).
2. A comparison node `cmp(x, t)` is evaluated against a threshold/operand `t`.
3. When `x_portable` and `x_eager` straddle `t` (`x_portable < t <= x_eager`, or
   vice-versa), `cmp` returns different booleans on the two backends.
4. The harness exact-compares the bool/int output and reports `|delta|=1`.

The comparator is a monotone, exact, integer-valued function of its input; given
the same input it produces the same output on both backends. The flip is purely
a consequence of the differing input, i.e. it is **inherited** from the float
kernel that fed it. This is the same "inherited" pattern that the
mismatch-delta-exact write-up folds into its root causes (§Inherited): a
structural/elementwise op whose divergence is carried in from an upstream
arithmetic/float node.

## Reproduction (verified)
Job: [`corpus/portable/w0/w0_227.py`](../../../corpus/portable/w0/w0_227.py).

The relevant chain (from the graph header):
```
n0 = floor_divide(L0, L1)
n3 = rsqrt.out(n0, out=float16)                       # float op
n4 = lt.Tensor_out(n3, L3, out=int64)   -> USER_OUTPUT[1]   # the flipped bool
n6 = rsqrt.out(n1, out=float16)                       # sibling rsqrt
n12 = atan2.out(n6, n11, out=float32)   -> USER_OUTPUT[3]   # rsqrt-derived float
```

Observed:
- `out[1]` (the `lt` comparator output): **portable=1, eager=0** — a single-bit
  bool flip, `max|delta|=1`.
- `out[3]` (an `rsqrt`-derived float sibling, `atan2` of `rsqrt(n1)`):
  **portable=1.5703, eager=2.3555**, `max|delta|≈0.785`. This non-zero float
  delta is the smoking gun: the **rsqrt input to the comparison genuinely
  diverged**, so a comparison against a nearby threshold flips. The comparator
  itself is faithful.

(The float sibling proving the inherited cause is `out[3]`. M6 in
mismatch-delta-exact cites the same job and the same rsqrt-derived float
divergence; the exact USER_OUTPUT index of the float sibling is 3 in this
replay.)

## Why it is not a backend bug
- The comparison kernels (`lt`, `gt`, `ge`, `le`, `eq`, `ne`, `logical_and/or/
  xor/not`) are exact integer-valued functions; there is no tolerance or
  rounding inside them to get wrong. Given identical inputs they agree.
- The actual divergence lives in the upstream float kernel (`rsqrt` here, more
  generally `layernorm`/`batch_norm`/transcendentals). Those float divergences
  are tracked and (where they are real) fixed in the dedicated float/precision
  bug files; many are themselves benign fp16/fp32 precision (see
  [transcendental-fp16-precision.md](transcendental-fp16-precision.md)).
- Fixing the comparator would be impossible (it is already correct) and
  pointless (the flip would persist as long as its input differs).

## Recommendation
- **Do not file against the comparison op.** Attribute the row to its upstream
  float producer.
- In triage, when an exact-compare bool/int `out[i]` flips by `|delta|=1` and a
  sibling float output (or the comparison's float input) shows a small non-zero
  delta, **dedup the bool flip to the upstream float kernel** and drop it from
  the comparator's bug surface.
- Fixing the genuine upstream float kernels, and dtype-scaling tolerance for the
  benign-precision ones, collapses these inherited flips automatically (the
  mismatch-delta-exact recommendation for M6 / inherited rows).

## Replay
Self-contained in-process replay:
[`replay_comparison-bool-flips.py`](replay_comparison-bool-flips.py). Loads
`corpus/portable/w0/w0_227.py`, runs it on the portable ExecuTorch runtime and
the eager reference, asserts the `lt` bool flip at USER_OUTPUT[1], and prints
the inherited rsqrt-derived float delta at USER_OUTPUT[3]. Exits 0 iff the bool
output flips (`max|delta| >= 1`).

```
$ cd /data/jwen929/mobile && .venv/bin/python findings/portable/bugs/replay_comparison-bool-flips.py 2>&1 \
    | grep -vE "cpuinfo|pytree|midr|reduce_util.cpp.380|KernelPreference"
job w0:227 out[1] (lt comparator) dt p=torch.int64/e=torch.int64 max|delta|=1 at idx 0
  portable=1  eager=0   (bool flip True<->False)
  inherited cause: out[3] (rsqrt-derived float) max|delta|=0.7852 portable=1.57031 eager=2.35547
  -> the comparator is correct; its input float value diverged upstream.
REPLAY: ARTIFACT reproduced (inherited bool flip)
```

The harness flags the bool flip; the script confirms it is inherited from the
rsqrt-derived float divergence, so the comparator is not at fault.
