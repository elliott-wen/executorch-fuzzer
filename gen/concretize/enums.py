"""String-enum argument tables.

Some native args accept only a small known set of strings (e.g. gelu's
`approximate` ∈ {none, tanh}). A slice constrains the arg's int var to a map's
keys so every generated value materialises a string the operator accepts.
"""

from __future__ import annotations


# Per-argument string enum maps (arg name → {int code: string}).  Used when a
# specific native argument only accepts a small known set of strings that the
# generic table below does not cover (e.g. gelu's `approximate` ∈ {none, tanh}).
# A slice/aigen constrains the arg's int var to this map's keys so every generated
# value materialises a string the operator actually accepts — letting the full
# valid space be reached instead of pinning to the single generic-table hit.
_STR_ENUM_BY_ARG = {
    "approximate": {0: "none", 1: "tanh"},   # gelu
    "UPLO": {0: "L", 1: "U"},                # linalg eigh/eigvalsh/cholesky/... (always L/U)
    "side": {0: "left", 1: "right"},         # searchsorted / bucketize
}

# Op-aware string-enum maps for args whose valid string set depends on the
# operator (arg-name alone collides).  Keyed by (op_base, arg_name) where
# op_base = op_name.split(".")[0].  Checked BEFORE the arg-name table.
#   - scatter.reduce / scatter.value_reduce use get_operator_enum(..., new=false)
#     → {add, multiply}.  (scatter_reduce.two / index_reduce use the new-options
#     {sum,prod,mean,amax,amin}; their aigens pin codes the generic table already
#     maps to valid strings, so they are deliberately NOT listed here.)
#   - linalg_qr `mode` ∈ {reduced, complete, r}.
_STR_ENUM_BY_OP_ARG = {
    ("scatter",   "reduce"): {0: "add", 1: "multiply"},
    ("linalg_qr", "mode"):   {0: "reduced", 1: "complete", 2: "r"},
    # pad mode ∈ {constant, reflect, replicate, circular} — NOT the interpolation
    # modes in the generic fallback (which caused "Unrecognised padding mode none").
    ("pad",       "mode"):    {0: "constant", 1: "reflect", 2: "replicate", 3: "circular"},
    # conv*d.padding / _convolution_mode `padding` string ∈ {valid, same}.
    ("conv1d",    "padding"): {0: "valid", 1: "same"},
    ("conv2d",    "padding"): {0: "valid", 1: "same"},
    ("conv3d",    "padding"): {0: "valid", 1: "same"},
    ("_convolution_mode", "padding"): {0: "valid", 1: "same"},
}


def _build_string_enum(value: int, arg_name: str | None = None,
                       op_name: str | None = None) -> str:
    # Op-aware first: (op_base, arg_name) overrides the generic arg-name table.
    if op_name is not None:
        op_base = op_name.split(".", 1)[0]
        per = _STR_ENUM_BY_OP_ARG.get((op_base, arg_name))
        if per is not None:
            return per.get(value, per[next(iter(per))])
    # Argument-aware: if the arg has a known valid string set, use it.
    per = _STR_ENUM_BY_ARG.get(arg_name)
    if per is not None:
        return per.get(value, per[next(iter(per))])
    # Generic fallback for unmapped string args.  This small stable table keeps
    # runtime verification moving; unsupported exact strings still become
    # inconclusive at the operator-call layer rather than at concretization.
    return {
        0: "none",
        1: "mean",
        2: "sum",
        3: "nearest",
        4: "linear",
        5: "bilinear",
        6: "bicubic",
    }.get(value, "none")
