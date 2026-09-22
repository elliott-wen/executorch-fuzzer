"""Op preconditions in Z3.

- model:   the Z3 model of ATen's type/shape system — TensorVar, ScalarVar, IntArrayVar,
           dtype promotion, broadcasting. The vocabulary every constraint is written in,
           kept byte-identical to the upstream model the constraints were generated
           against.
- symbols: mangled C++ symbol → candidate ATen op names. BUILD-TIME ONLY — the resolved
           answer lives in ops/table_data.py; nothing here reads this at run time.
- loader:  lazy, hermetic loading of one op's constraints from pytorch_constraints/.

Typical use — narrow by name first, load only what survives:

    from mobile.generator.constraints import load
    from mobile.generator.ops import symbols

    for symbol in symbols():                   # the table rows worth loading
        oc = load(symbol)                       # exec'd here, on demand

Everything here describes EAGER: when PyTorch itself raises. What a particular runtime
additionally refuses lives in generator/targets, which composes the two — see its load_for.
"""

from __future__ import annotations

from .loader import OpConstraints, available, load, root, set_root

__all__ = ["OpConstraints", "available", "load", "root", "set_root"]
