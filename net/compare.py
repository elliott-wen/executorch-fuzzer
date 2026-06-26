"""compare.py — the single owner of eager-vs-ExecuTorch output comparison + reporting.

Self-contained (no generate/execute imports). Runs in the FEEDER, which holds the eager
reference (it read the corpus) and receives the worker's raw outputs back via the broker —
so the diff happens HERE, and the broker stays a pure router that never touches a tensor.

  compare(eager, et)        -> (status, detail)   the verdict (OK / MISMATCH / SKIP)
  select(et, user_pos)      -> et                 keep the USER_OUTPUT graph outputs
  report(eager, et)         -> [per-output dict]  rich detail for single-graph debugging

`compare` returns (status, detail): OK if equal, MISMATCH if they differ. Two things make the
comparison robust to ExecuTorch's output conventions rather than reporting spurious
mismatches:
  • ExecuTorch PREPENDS user-input-mutation outputs (a mutated graph input — e.g. an
    out=/indices= port wired to a leaf), so the real graph outputs are the TRAILING ones.
  • non-finite values must match position-for-position (nan==nan AND inf==inf): plain
    allclose flags |inf-inf|=nan as unequal, so matching infinities looked like a diff.
"""

from __future__ import annotations

import torch

# One loose tolerance for ALL floating-point / complex comparisons — fp16 and fp32
# round differently from eager's per-op rounding, so anything tighter just reports
# precision noise. Integer/bool stay EXACT (tol 0): an integer divergence is a real bug
# (e.g. sum/prod cast-order off-by-one, bitwise) and a relative tolerance would hide it.
#
# QUANTIZED backends: pregen stores a QUANTIZED reference (the PT2E-converted graph run on
# CPU — see gen/backends/base.quantized_reference), NOT the fp32 eager oracle. So this
# tolerance is applied to device-vs-quantized-reference (same quantization space): the
# residual is cross-implementation int rounding, not raw int8-vs-fp32 error. For a CPU
# delegate (xnnpack) that residual is ~0; for an NPU (Ethos-U on the FVP) it may need a
# looser, per-backend tolerance once real device deltas are observed.
_FLOAT_RTOL, _FLOAT_ATOL = 1e-2, 1e-3


def _tol(dtype):
    if dtype.is_floating_point or dtype.is_complex:
        return _FLOAT_RTOL, _FLOAT_ATOL
    return 0.0, 0.0


def _cmp(a, b, i):
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return "SKIP", "out[%d] non-tensor" % i
    if a.dtype != b.dtype:
        return "MISMATCH", "out[%d] dtype %s vs %s" % (i, a.dtype, b.dtype)
    if a.shape != b.shape:
        return "MISMATCH", "out[%d] shape %s vs %s" % (i, tuple(a.shape), tuple(b.shape))

    af, bf = a.float(), b.float()
    # Non-finite structure must agree element-wise (nan, +inf, -inf positions). This makes
    # matching nan/inf EQUAL — and still flags a genuine nan-vs-inf (or finite-vs-nan).
    if not (torch.equal(af.isnan(), bf.isnan())
            and torch.equal(af == float("inf"), bf == float("inf"))
            and torch.equal(af == float("-inf"), bf == float("-inf"))):
        return "MISMATCH", "out[%d] non-finite mismatch (nan/inf positions differ)" % i

    rtol, atol = _tol(a.dtype)
    finite = af.isfinite() & bf.isfinite()      # identical mask now; compare only here
    if finite.any():
        diff = (af[finite] - bf[finite]).abs()
        if bool((diff > atol + rtol * bf[finite].abs()).any()):
            return "MISMATCH", ("out[%d] max|delta|=%.3e (rtol=%s atol=%s)"
                                % (i, diff.max().item(), rtol, atol))
    return "OK", ""


def compare(eager: list, et: list) -> tuple[str, str]:
    """Compare eager reference outputs vs ExecuTorch outputs → (status, detail)."""
    # Drop ExecuTorch's leading user-input-mutation outputs; the graph outputs are the
    # trailing len(eager) (the eager reference never includes mutated inputs).
    if len(et) > len(eager):
        et = et[len(et) - len(eager):]
    if len(eager) != len(et):
        return "MISMATCH", "output count %d vs %d" % (len(eager), len(et))

    status, detail = "OK", ""
    for i, (a, b) in enumerate(zip(eager, et)):
        s, d = _cmp(a, b, i)
        if s == "MISMATCH":
            return s, d
        if s == "SKIP":
            status, detail = s, d
    return status, detail


def select(et: list, user_pos: list | None) -> list:
    """Keep only the USER_OUTPUT graph outputs (`user_pos`), dropping the mutated-input aliases
    (out=/indices= buffers wired to a leaf) ExecuTorch also returns, so the diff lines up with eager.
    A missing/out-of-range user_pos falls back to all outputs (compare() then trims the leading ones)."""
    if user_pos is not None and all(0 <= i < len(et) for i in user_pos):
        return [et[i] for i in user_pos]
    return et


def _samples(af, bf, d):
    n = af.numel()
    if n == 0:
        return [], []
    idxs = list(range(min(n, 8)))
    if n > 8:
        j = int(d.argmax())                          # always include the worst element
        if j not in idxs:
            idxs[-1] = j
    return ([round(float(af[k]), 6) for k in idxs],
            [round(float(bf[k]), 6) for k in idxs])


def report(eager: list, et: list) -> list:
    """Per-output detail for a single-graph diff (the debug observation): dtype/shape, max|delta|,
    and a few aligned eager/et samples (the worst element always included)."""
    if len(et) > len(eager):
        et = et[len(et) - len(eager):]
    recs = []
    for i, (a, b) in enumerate(zip(eager, et)):
        r = {"idx": i, "dtype": str(a.dtype).replace("torch.", ""), "shape": list(a.shape)}
        if isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype:
            af, bf = a.float().flatten(), b.float().flatten()
            d = (af - bf).abs()
            finite = af.isfinite() & bf.isfinite()
            if finite.any():
                r["max_abs_delta"] = float(d[finite].max())
            r["eager"], r["et"] = _samples(af, bf, d)
        else:                                        # dtype/shape divergence — the bug itself
            r["et_dtype"] = str(getattr(b, "dtype", "?")).replace("torch.", "")
            r["et_shape"] = list(getattr(b, "shape", []) or [])
        recs.append(r)
    return recs
