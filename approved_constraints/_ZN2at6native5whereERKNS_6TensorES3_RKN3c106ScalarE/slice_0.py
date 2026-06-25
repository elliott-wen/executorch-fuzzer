import sys; sys.path.insert(0, '/data/jwen929/pytorch/z3')
from z3 import *
from model import TensorVar, ScalarVar, IntArrayVar, OptVar, TensorListVar, expandable_to, broadcastable, same_shape

ENTRY = '_ZN2at6native5whereERKNS_6TensorES3_RKN3c106ScalarE (aigen)'
IN_LOOP = False
LOOP_SAFE = True
_UNRESOLVED = frozenset([])
def get_unresolved_vars(): return _UNRESOLVED

_PARAMS = [('condition', 'Tensor'), ('self', 'Tensor'), ('other', 'Scalar')]
def get_params(): return _PARAMS

def get_constraints():
    condition = TensorVar('condition')
    self_ = TensorVar('self')
    other = ScalarVar('other')
    # C++: TensorCompare.cpp:658 — where(cond, Tensor self, Scalar other);
    #   other -> scalar_tensor (0-dim), broadcasts against everything.
    # C++: TensorCompare.cpp:626 — condition must be kBool (Byte=0 deprecated/auto-cast).
    cond_bool = Or(condition.dtype == 11, condition.dtype == 0)
    # C++: TensorCompare.cpp:631-637 — TensorIterator broadcasts condition with self.
    bcast = broadcastable(condition, self_)
    valid = And(cond_bool, bcast)
    return [Not(valid)]
