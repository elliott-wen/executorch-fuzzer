"""concretize — a satisfying Z3 model → concrete PyTorch call arguments.

Given an op's params, its model variables, and a model from the solver, this materialises
the actual tensors/scalars/lists to call the op with, in signature order.

It is a faithful translator: it reports what the solver decided and adds no policy of its
own. In particular it does NOT rewrite dtypes — changing a solved dtype here contradicts a
relation the solver had just satisfied (canCast, promote_types, same-as between two
arguments), leaving the op with inputs that violate its own precondition. A runtime that
needs narrower dtypes says so in generator/targets, where the solver satisfies it along
with everything else.

Package layout:
  dtypes   — model ScalarType code → torch.dtype maps
  evals    — Z3 model → Python scalar evaluators (_eval_*)
  enums    — string-enum argument tables (_build_string_enum)
  builders — per-type value builders (_build_tensor/_build_scalar/_build_value/...)
  coerce   — special-arg coercion (_coerce_special_arg)
  args     — build_call_args entry point + ConcreteArgs
"""

from __future__ import annotations

from .args import build_call_args, ConcreteArgs
from .dtypes import _torch_dtypes, dtype_code_for

__all__ = ["build_call_args", "ConcreteArgs", "_torch_dtypes", "dtype_code_for"]
