"""Supplementary guard slice for bitwise_right_shift(Tensor, Scalar).

The combined slice_0 lists BOOL (dtype 11) among the allowed `self` dtypes, but
eager PyTorch has no shift kernel for bool:
    RuntimeError: "rshift_cpu" not implemented for 'Bool'
This slice's constraint is a FAILURE path (self is bool); the harness adds
Not(And(get_constraints())), i.e. `self.dtype != BOOL`, excluding it.
Hand-authored guard; kept in a separate slice so it survives a slice_0 recombine."""

import sys
sys.path.insert(0, '/data/jwen929/pytorch/z3')
from z3 import *
from model import TensorVar, ScalarVar, IntArrayVar, OptVar, TensorListVar

ENTRY = '_ZN2at6native19bitwise_right_shiftERKNS_6TensorERKN3c106ScalarE (bool guard)'
IN_LOOP = False
LOOP_SAFE = True
_UNRESOLVED = frozenset([])

BOOL = 11


def get_unresolved_vars() -> frozenset:
    return _UNRESOLVED


_PARAMS = [('self', 'Tensor'), ('other', 'Scalar')]


def get_params() -> list:
    return _PARAMS


def get_constraints() -> list:
    # Failure region: self is bool. Harness negates -> self.dtype != BOOL.
    return [Int('self.dtype') == IntVal(BOOL)]


def get_axioms() -> list:
    return []


def get_vars() -> dict:
    self = TensorVar('self')
    return {'self': self}


def build_solver() -> Solver:
    s = Solver()
    s.add(*get_axioms())
    return s
