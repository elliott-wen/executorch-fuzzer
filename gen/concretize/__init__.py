"""concretize — MOBILE-LOCAL copy of z3/concretize, tweakable for the ExecuTorch
path independently of the main pipeline.

Converts a Z3 model into concrete PyTorch call arguments. The mobile pre-gen path
injects this copy into the generator via gen.ops.opnode.set_build_call_args, so
graphs for the ExecuTorch fuzzer are concretized HERE — tweak dtype/shape
selection, value ranges, etc. (e.g. to bias toward what the portable runtime
supports) without touching the canonical z3/concretize.py.

Still shares model.py (the symbolic constraint model) with the main pipeline; only
the concretization logic is forked.

Package layout:
  dtypes   — dtype maps + simplification (_torch_dtypes, _simplify_dtype)
  evals    — Z3 model → Python scalar evaluators (_eval_*)
  enums    — string-enum argument tables (_build_string_enum)
  builders — per-type value builders (_build_tensor/_build_scalar/_build_value/...)
  coerce   — special-arg coercion (_coerce_special_arg)
  args     — build_call_args entry point + ConcreteArgs
"""

from __future__ import annotations

from .args import build_call_args, ConcreteArgs
from .dtypes import _torch_dtypes

__all__ = ["build_call_args", "ConcreteArgs", "_torch_dtypes"]
