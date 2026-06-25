import sys; sys.path.insert(0, '/data/jwen929/pytorch/z3')
from z3 import *
from model import TensorVar, ScalarVar, IntArrayVar, OptVar, TensorListVar, expandable_to, broadcastable, same_shape

ENTRY = '_ZN2at6native5whereERKNS_6TensorERKN3c106ScalarES7_ (aigen)'
IN_LOOP = False
LOOP_SAFE = True
_UNRESOLVED = frozenset([])
def get_unresolved_vars(): return _UNRESOLVED

_PARAMS = [('condition', 'Tensor'), ('self', 'Scalar'), ('other', 'Scalar')]
def get_params(): return _PARAMS

def get_constraints():
    condition = TensorVar('condition')
    self_ = ScalarVar('self')
    other = ScalarVar('other')
    # C++: TensorCompare.cpp:666 — where(cond, Scalar self, Scalar other);
    #   both scalars -> scalar_tensor (0-dim), broadcast against condition freely.
    # C++: TensorCompare.cpp:626 — condition must be kBool (Byte=0 deprecated/auto-cast).
    cond_bool = Or(condition.dtype == 11, condition.dtype == 0)
    valid = cond_bool
    return [Not(valid)]
