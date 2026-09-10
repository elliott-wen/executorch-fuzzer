# XNNPACK-x86 bugs — one file + buggy model graph each

The bugs introduced by the **XNNPACK delegate** (everything else in the xnnpack run is shared with
portable — see [../README.md](../README.md)). Each bug has a focused md and a co-located **buggy
model graph** `graph_<bug>.py` — the representative corpus graph that triggers it.

Unlike the [ARM bugs](../../xnnpack-arm/bugs/README.md) (NEON-only, need the phone), these reproduce
**in-process on the x86 host**: the x86 SSE/AVX min/max + fast-approximation microkernels are what
launder the non-finite values. Running a `graph_*.py` lowers it through the XNNPACK delegate and runs
the **first-divergence localizer** (`mobile.gen.divergence`, `localize(..., backend="xnnpack")`) — no
broker or client/worker needed. Run from the repo root, e.g.
`.venv/bin/python findings/xnnpack_x64/bugs/graph_xnnpack-clamp-minmax-nan.py`.

> ⚠ **Authored but NOT re-run this round** — no xnnpack host client was available to verify, so the
> "expected" line in each graph/md is the localizer-confirmed result documented during the original
> analysis, not a fresh capture. (Contrast the host [portable replays](../../portable/REPLAY.md),
> which were executed.)

| bug | class | verdict | graph |
|---|---|---|---|
| [xnnpack-clamp-minmax-nan.md](xnnpack-clamp-minmax-nan.md) | **real XNNPACK correctness** (NaN launder) | MISMATCH (non-finite) | [`graph`](graph_xnnpack-clamp-minmax-nan.py) (`w0:49`) |
| [xnnpack-sqrt-rsqrt-domain.md](xnnpack-sqrt-rsqrt-domain.md) | **real XNNPACK correctness** (domain) | MISMATCH (non-finite) | [`graph`](graph_xnnpack-sqrt-rsqrt-domain.py) (`w0:963`) |
| [xnnpack-exp-nonfinite.md](xnnpack-exp-nonfinite.md) | **real XNNPACK correctness** (no guard) | MISMATCH (non-finite) | [`graph`](graph_xnnpack-exp-nonfinite.py) (`w0:938`) |
| [xnnpack-load-failure.md](xnnpack-load-failure.md) | **real XNNPACK partition over-inclusion** | SKIP (`.pte` unloadable) | [`graph`](graph_xnnpack-load-failure.py) (`w0:1175`) |

**Headline:** the activation/elementwise non-finite cluster (clamp/min/max NaN-launder, sqrt/rsqrt
domain, exp no-guard) is the largest new family — XNNPACK's SIMD min/max + fast-approx kernels
silently turn `NaN`/`sqrt(neg)`/`exp(nan)` into finite values where eager and portable propagate the
non-finite. The load-failure is a separate partition over-inclusion bug: the partitioner delegates
rank-7 (`pixel_shuffle`) tensors past `XNN_MAX_TENSOR_DIMS=6`, making the whole `.pte` unloadable.
The first three were split out of the former `xnnpack-activation-nonfinite.md` (now a
[stub index](xnnpack-activation-nonfinite.md)).

Shared (non-XNNPACK-specific) bugs are not duplicated here — the crashes and most mismatches/skips
fall back to the **same portable kernels**; see [../crash/](../crash/README.md),
[../skip/](../skip/README.md) and [../../portable/bugs/](../../portable/bugs/README.md).

Trigger helper: [`_trigger.py`](_trigger.py) (imported by each `graph_*.py` `__main__`): runs the
host XNNPACK first-divergence localizer on the graph and prints the born-here divergence (or, for the
load-failure bug, the load error).
