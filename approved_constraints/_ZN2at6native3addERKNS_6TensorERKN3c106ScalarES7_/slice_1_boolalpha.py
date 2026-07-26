"""Supplementary guard slice for add.Scalar(Tensor self, Scalar other, Scalar alpha).

The combined slice_0 only enforces `alpha bool => self bool`, but eager PyTorch
requires the boolean alpha to match the RESULT dtype, and add.Scalar's result is
bool only when BOTH self and the scalar `other` are bool. With e.g.
    add.Scalar(bool_tensor, -1, alpha=True)
self is bool but `other` is an int scalar, so the result promotes to int and:
    RuntimeError: Boolean alpha only supported for Boolean results.
This slice's constraint is the FAILURE path (alpha is bool while the result is
not); the harness adds Not(And(...)), i.e. `alpha bool => (self bool AND other
bool)`. Hand-authored guard; separate slice so it survives a slice_0 recombine."""

import sys
sys.path.insert(0, '/data/jwen929/pytorch/z3')
from z3 import *
from model import TensorVar, ScalarVar, IntArrayVar, OptVar, TensorListVar

ENTRY = '_ZN2at6native3addERKNS_6TensorERKN3c106ScalarES7_ (bool-alpha guard)'
IN_LOOP = False
LOOP_SAFE = True
_UNRESOLVED = frozenset([])

BOOL = 11


def get_unresolved_vars() -> frozenset:
    return _UNRESOLVED


_PARAMS = [('self', 'Tensor'), ('other', 'Scalar'), ('alpha', 'Scalar')]


def get_params() -> list:
    return _PARAMS


def get_constraints() -> list:
    # Failure region: alpha is bool but the result is not bool (self and other
    # are not both bool). Harness negates -> alpha bool => (self bool & other bool).
    return [And(
        Int('alpha.dtype') == IntVal(BOOL),
        Not(And(Int('self.dtype') == IntVal(BOOL),
                Int('other.dtype') == IntVal(BOOL))),
    )]


def get_axioms() -> list:
    return []


def get_vars() -> dict:
    self = TensorVar('self')
    other = ScalarVar('other')
    alpha = ScalarVar('alpha')
    return {'self': self, 'other': other, 'alpha': alpha}


def build_solver() -> Solver:
    s = Solver()
    s.add(*get_axioms())
    return s
