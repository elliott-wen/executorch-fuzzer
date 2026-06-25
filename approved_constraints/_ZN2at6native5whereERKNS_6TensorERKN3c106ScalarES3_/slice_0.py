import sys; sys.path.insert(0, '/data/jwen929/pytorch/z3')
from z3 import *
from model import TensorVar, ScalarVar, IntArrayVar, OptVar, TensorListVar, expandable_to, broadcastable, same_shape

ENTRY = '_ZN2at6native5whereERKNS_6TensorERKN3c106ScalarES3_ (aigen)'
IN_LOOP = False
LOOP_SAFE = True
_UNRESOLVED = frozenset([])
def get_unresolved_vars(): return _UNRESOLVED

_PARAMS = [('condition', 'Tensor'), ('self', 'Scalar'), ('other', 'Tensor')]
def get_params(): return _PARAMS

def get_constraints():
    condition = TensorVar('condition')
    self = ScalarVar('self')
    other = TensorVar('other')
    # C++: TensorCompare.cpp:650 where.ScalarSelf -> at::where(cond, self_t, other_converted)
    #      -> where_self_out: TensorCompare.cpp:626 TORCH_CHECK(condition.scalar_type()==kBool)
    #      Byte(uint8) is silently accepted (line 621-625, deprecated warn -> cast to bool).
    #      => condition.dtype in {Bool=11, Byte=0}
    # broadcast: condition must be broadcastable with other (TensorIterator infer_size).
    valid = And(
        Or(condition.dtype == 11, condition.dtype == 0),
        broadcastable(condition, other),
    )
    return [Not(valid)]
