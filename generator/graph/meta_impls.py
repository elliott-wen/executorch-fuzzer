"""meta_impls.py — meta rules for ops the runtime doesn't ship one for.

Every candidate node is validated by running it on meta tensors: shape and dtype only, no
data, no kernel. That is what keeps generation cheap and keeps a broken kernel from taking
the generation worker down. A few ops have no meta rule at all, so the probe fails and the
op can never enter a graph — `_native_batch_norm_legit.out` is worse than that, it
segfaults the process outright when run for real instead.

Supplying the rule ourselves fixes that, and it is exact wherever the output shape is a
pure function of the INPUT shapes.

It is NOT a general escape hatch. `nonzero` and `masked_select` derive their output shape
from the input's VALUES, which meta tensors do not carry, so no rule written here can be
correct — the honest upper bound (every element selected) is wrong for almost every actual
input. It would still let those ops generate, but only as graph SINKS: anything wired
downstream would be pinned to a shape the graph does not produce at run time, and would
fail at the eager reference. Supporting them properly means teaching the builder to never
anchor onto a data-dependent producer, which is a separate change from this file.
"""

from __future__ import annotations

import torch

_LIB = torch.library.Library("aten", "IMPL")
_registered = False


def _batch_norm_legit_out(input, weight, bias, running_mean, running_var,
                          training, momentum, eps, out, save_mean, save_invstd):
    """`out` takes the input's shape; the saved statistics are per-channel, and empty
    when not training (ATen skips computing them on the inference path)."""
    channels = input.size(1) if input.dim() > 1 else 0
    stats = (channels,) if training else (0,)
    return (input.new_empty(input.shape),
            input.new_empty(stats),
            input.new_empty(stats))


def _batch_norm_legit_no_stats_out(input, weight, bias, training, momentum, eps,
                                   out, save_mean, save_invstd):
    """Same rule without the running statistics arguments."""
    channels = input.size(1) if input.dim() > 1 else 0
    stats = (channels,) if training else (0,)
    return (input.new_empty(input.shape),
            input.new_empty(stats),
            input.new_empty(stats))


_IMPLS = {
    "_native_batch_norm_legit.out": _batch_norm_legit_out,
    "_native_batch_norm_legit.no_stats_out": _batch_norm_legit_no_stats_out,
}


def register() -> None:
    """Install the rules. Idempotent — torch raises if the same key is registered twice."""
    global _registered
    if _registered:
        return
    for name, fn in _IMPLS.items():
        _LIB.impl(name, fn, "Meta")
    _registered = True
