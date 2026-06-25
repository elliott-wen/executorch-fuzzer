"""blocklist.py — ops to EXCLUDE from all-ops mode.

In all-ops mode we want *most* of the ~1343 approved ops; we only block a few
that are known to produce noise or unsafe results rather than legitimate
findings. `is_blocked(op_name)` decides per resolved op_name (e.g. 'add.Tensor',
'_reshape_alias.default', 'mm.out').

Categories:
  - Unsafe view/alias primitives: trust the caller to pass numel-preserving args;
    composing them with arbitrary shapes fabricates corrupt tensors (garbage-in),
    not real bugs. (e.g. _reshape_alias([0]->[]) → malformed scalar → downstream OOB)
  - Mutating/aliasing storage ops: break the clean-DAG assumption.
  - Random/sampling ops: their result depends on a runtime RNG draw, and eager vs
    ExecuTorch don't share an RNG stream — so identical inputs still diverge and
    produce false MISMATCHes. Matched two ways: a PREFIX (catches whole families
    like normal/normal_/normal_functional) and a BASE-NAME set _RNG_BASES (catches
    stragglers whose name doesn't start with an RNG prefix, e.g. native_dropout).
Edit BLOCK_PREFIXES / BLOCK_EXACT / _RNG_BASES to tune what the sweep covers.
"""

from __future__ import annotations

# Block if the op_name starts with any of these (matches all overloads).
BLOCK_PREFIXES: tuple[str, ...] = (
    # ── unsafe view / alias / storage primitives ──
    "_reshape_alias",       # unsafe view: no numel check → corrupt tensor
    "set_",                 # rebinds storage → malformed tensor
    "set.",
    "as_strided",           # arbitrary strides/offset → OOB views
    "_unsafe_view",
    "unsafe_",
    "view_copy.dtype",      # dtype-reinterpret view → element-size mismatch crashes
    "view.dtype",
    "resize_",              # resizes storage in place
    "_resize_output",
    # ── random / sampling families (prefix-matched; base-name stragglers below) ──
    "rand",                 # rand, randn, randint, randperm, rand_like, randn_like, ...
    "random",               # random_, random.*
    "normal",               # normal, normal_, normal_functional
    "bernoulli",
    "poisson",
    "multinomial",
    "uniform",              # uniform_
    "exponential",          # exponential_
    "cauchy",               # cauchy_
    "geometric",            # geometric_
    "log_normal",           # log_normal_
    "_sample_dirichlet",
    "dropout",              # nondeterministic mask
    "_fused_dropout",
)

# RNG ops whose name does NOT start with an RNG prefix above, so prefix-matching
# misses them — matched on the BASE name (op_name before the first '.').
# e.g. native_dropout (starts "native"), rrelu / rrelu_with_noise.
_RNG_BASES: frozenset[str] = frozenset({
    "native_dropout", "rrelu", "rrelu_with_noise",
})

# Block these exact op_names.
BLOCK_EXACT: frozenset[str] = frozenset({
})


def is_blocked(op_name: str | None) -> bool:
    if not op_name:
        return True
    if op_name in BLOCK_EXACT:
        return True
    if op_name.split(".", 1)[0] in _RNG_BASES:   # base-name RNG stragglers
        return True
    return any(op_name.startswith(p) for p in BLOCK_PREFIXES)
