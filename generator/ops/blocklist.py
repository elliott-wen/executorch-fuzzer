"""blocklist.py — ops to EXCLUDE from all-ops mode.

All-ops mode sweeps *most* of the ~1343 approved ops; this module carves out the few
that yield noise or unsafe garbage instead of real findings. `is_blocked(op_name)` is
the single entry point, called per resolved op_name (e.g. 'add.Tensor',
'_reshape_alias.default', 'mm.out').

A name is blocked if ANY of three matchers fires:
  1. BLOCK_EXACT    — exact op_name match (overload-specific).
  2. _RNG_BASES     — match on the BASE name (before the first '.'); catches RNG ops
                      whose name doesn't start with an RNG prefix (e.g. native_dropout).
  3. BLOCK_PREFIXES — startswith match; blocks a whole op family and all its overloads.

BLOCK_PREFIXES is assembled from the per-category groups below; each group carries the
rationale for why those ops are blocked. Edit the relevant group to tune coverage.
"""

from __future__ import annotations

# ── Unsafe view / alias / storage primitives ────────────────────────────────────
# These trust the caller to pass numel-preserving / in-bounds args; composing them with
# arbitrary fuzzer shapes fabricates corrupt tensors (garbage-in), not real bugs — e.g.
# _reshape_alias([0]->[]) → malformed scalar → downstream OOB. The mutating/aliasing
# storage ops (set_/resize_) also break the clean-DAG assumption the generator relies on.
_UNSAFE_VIEW_PREFIXES: tuple[str, ...] = (
    "_reshape_alias",   # unsafe view: no numel check → corrupt tensor
    "set_",             # rebinds storage → malformed tensor
    "set.",
    "as_strided",       # arbitrary strides/offset → OOB views
    "_unsafe_view",
    "unsafe_",
    "view_copy.dtype",  # dtype-reinterpret view → element-size mismatch crashes
    "view.dtype",
    "resize_",          # resizes storage in place
    "_resize_output",
)

# ── Random / sampling families ──────────────────────────────────────────────────
# Result depends on a runtime RNG draw, and eager vs ExecuTorch don't share an RNG
# stream — so identical inputs still diverge and produce false MISMATCHes. Prefix-matched
# here (catches whole families like normal/normal_/normal_functional); stragglers whose
# name doesn't start with an RNG prefix are handled by _RNG_BASES below.
_RNG_PREFIXES: tuple[str, ...] = (
    "rand",             # rand, randn, randint, randperm, rand_like, randn_like, ...
    "random",           # random_, random.*
    "normal",           # normal, normal_, normal_functional
    "bernoulli",
    "poisson",
    "multinomial",
    "uniform",          # uniform_
    "exponential",      # exponential_
    "cauchy",           # cauchy_
    "geometric",        # geometric_
    "log_normal",       # log_normal_
    "_sample_dirichlet",
    "dropout",          # nondeterministic mask
    "_fused_dropout",
)

# ── Data-dependent output shape (cannot be shape-inferred) ──────────────────────
# These compute their output SHAPE from the input's VALUES, not from input shapes: nonzero
# returns one row per non-zero element, masked_select one element per set mask bit,
# repeat_interleave.Tensor a length that is the SUM of its repeats argument.
#
# Every candidate node is validated on meta tensors, which carry shape and dtype but no
# data, so there is nothing for these to derive a shape from — torch itself refuses ("the
# register_meta function for torch.nonzero() raises unimplemented by default"). Writing
# our own meta rule (see meta_impls.py) does not rescue them either: any shape we declare
# is a guess, and the emitted script regenerates its leaves from a seed, so probe-time
# data never matches run-time data. Anything wired downstream would be pinned to a shape
# the graph does not produce.
#
# They are only sound as graph SINKS. Supporting that means teaching the builder never to
# anchor onto a data-dependent producer; until then they are excluded rather than left to
# fail every attempt.
_DATA_DEPENDENT_SHAPE_PREFIXES: tuple[str, ...] = (
    "nonzero",                   # (nnz, ndim) — depends on how many elements are non-zero
    "masked_select",             # (popcount(mask),)
    "repeat_interleave.Tensor",  # (sum(repeats),); the int overloads are static and fine
)

# Block if op_name starts with any of these (matches every overload of the family).
BLOCK_PREFIXES: tuple[str, ...] = (
    *_UNSAFE_VIEW_PREFIXES,
    *_RNG_PREFIXES,
    *_DATA_DEPENDENT_SHAPE_PREFIXES,
)

# RNG ops whose name does NOT start with an RNG prefix above (so prefix-matching misses
# them) — matched on the BASE name, i.e. op_name before the first '.'. e.g. native_dropout
# (starts "native"), rrelu / rrelu_with_noise.
_RNG_BASES: frozenset[str] = frozenset({
    "native_dropout", "rrelu", "rrelu_with_noise",
})

# Exact op_names to block (overload-specific). Currently none.
BLOCK_EXACT: frozenset[str] = frozenset()


def is_blocked(op_name: str | None) -> bool:
    """True if `op_name` should be excluded from the all-ops sweep.

    A missing/empty name counts as blocked. Otherwise a name is blocked when it is in
    BLOCK_EXACT, its base name is in _RNG_BASES, or it starts with any BLOCK_PREFIXES entry.
    """
    if not op_name:
        return True
    base_name = op_name.split(".", 1)[0]
    return (
        op_name in BLOCK_EXACT
        or base_name in _RNG_BASES
        or op_name.startswith(BLOCK_PREFIXES)   # startswith accepts a tuple of prefixes
    )
