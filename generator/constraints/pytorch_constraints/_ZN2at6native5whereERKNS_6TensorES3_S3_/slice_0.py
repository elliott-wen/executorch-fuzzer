from z3 import *
from mobile.generator.constraints.model import TensorVar, ScalarVar, IntArrayVar, OptVar, TensorListVar, expandable_to, broadcastable, same_shape

ENTRY = '_ZN2at6native5whereERKNS_6TensorES3_S3_ (aigen)'
IN_LOOP = False
LOOP_SAFE = True
_UNRESOLVED = frozenset([])
def get_unresolved_vars(): return _UNRESOLVED

_PARAMS = [('condition', 'Tensor'), ('self', 'Tensor'), ('other', 'Tensor')]
def get_params(): return _PARAMS

def get_constraints():
    condition = TensorVar('condition')
    self_ = TensorVar('self')
    other = TensorVar('other')
    # C++: TensorCompare.cpp:626 — TORCH_CHECK(condition_.scalar_type()==kBool);
    #      uint8 (Byte=0) is accepted (deprecated, auto-cast to Bool at :624).
    cond_bool = Or(condition.dtype == 11, condition.dtype == 0)
    # C++: TensorCompare.cpp:631-637 — TensorIterator broadcasts condition/self/other.
    bcast = And(broadcastable(condition, self_),
                broadcastable(condition, other),
                broadcastable(self_, other))
    valid = And(cond_bool, bcast)
    return [Not(valid)]
