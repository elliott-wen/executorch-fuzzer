# Ruled out: non-finite mismatches are reference-side (3d/3e)

**433 delegated "nonfinite mismatch" samples** — `native_group_norm` (176), `gelu.out` (82),
`native_layer_norm` (77), `_softmax.out` (55), `div.out_mode` (19), `rsqrt.out` (14), `glu.out` (2),
`linear.out` (2), `mul.Scalar`/`linear`/`roll` (1 each).

**Filter:** a real device non-finite bug needs eager **finite** → device nan/inf, on a **finite** leaf.
A 30-sample device probe across these ops found **100% have a non-finite _eager_ reference**:

- `native_group_norm`, `rsqrt.out`: eager = **all-NaN** (degenerate reference), device = **finite**. The
  device is *more* correct than the CPU reference — opposite of a device bug.
- `gelu.out`, `native_layer_norm`, `_softmax.out`: eager = NaN, device = inf — both non-finite, driven
  by a NaN/inf-**injected leaf input** (the corpus domain-fuzz). A nan-vs-inf disagreement on garbage
  input, not "finite in → nonfinite out".

Per steps 3d (reference confound) and 3e (leaf already non-finite), **none of these are device
operator bugs.** (The genuine `native_group_norm` value bug is separate — finite eager, large delta —
and IS filed, see `../bugs/native_group_norm.md`.)
