# XNNPACK-ARM bugs — one file + buggy model graph each

Each bug below has a focused md and a co-located **buggy model graph** `graph_<bug>.py` — the
minimal (vlog) or representative-corpus (gelu/tan/erf/crash) graph that triggers it. Because
these are **NEON-only** bugs (they don't reproduce on the x86 host), running a graph lowers it to
the XNNPACK delegate and dispatches it to the **connected ARM phone** through the broker, then
diffs the device output vs eager. (Contrast the host `replay_*.py` under `../../portable/bugs/`,
which run a stored `.pte` in-process via `et_runner` — the host reproduces those.) **Prereq:** a
broker is running (`python broker.py`) and the ARM phone is attached as a worker. Run from the
repo root, e.g. `.venv/bin/python findings/xnnpack-arm/bugs/graph_neon-vlog-special-values.py`.

| bug | class | verdict | graph |
|---|---|---|---|
| [neon-vlog-special-values.md](neon-vlog-special-values.md) | **real NEON correctness** | MISMATCH (non-finite) | [`graph`](graph_neon-vlog-special-values.py) (minimal, 1 op) |
| [neon-gelu-nan-launder.md](neon-gelu-nan-launder.md) | **real NEON correctness** | MISMATCH (non-finite) | [`graph`](graph_neon-gelu-nan-launder.py) |
| [neon-tan-erf-precision.md](neon-tan-erf-precision.md) | NEON precision (fast-math) | MISMATCH (value Δ) | [`graph`](graph_neon-tan-erf-precision.py) |
| [arm-only-logsoftmax-version-crash.md](arm-only-logsoftmax-version-crash.md) | version confound (not a NEON bug) | CRASH | [`graph`](graph_arm-only-logsoftmax-version-crash.py) |

**Headline:** the one cleanly-arch, high-severity bug is the **NEON `f32-vlog` no-special-value**
defect (`log(+Inf)=88.376266`), which also drives the dominant `logit` cluster. `gelu` is a
separate NEON NaN→0 launder; `tan`/`erf` are reduced-precision drift; the `_log_softmax` "ARM-only
crash" is a build-version artifact, not an arch bug.

Shared (non-ARM-specific) bugs are not duplicated here — see
[../../xnnpack/bugs/](../../xnnpack/bugs/README.md) and [../../portable/bugs/](../../portable/bugs/README.md).

Trigger helper: `_trigger.py` (imported by each `graph_*.py` `__main__`): lowers the graph to
XNNPACK via `build_job`, dispatches it to the phone, and prints device-vs-eager at each diverging
output. The minimal vlog graph is a clean 1-op repro; gelu/tan/erf/crash use the representative
corpus graph because single-op isolation isn't delegated by the XNNPACK partitioner.
