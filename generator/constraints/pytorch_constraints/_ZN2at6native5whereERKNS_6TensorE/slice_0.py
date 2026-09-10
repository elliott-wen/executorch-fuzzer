from z3 import *
from mobile.generator.constraints.model import TensorVar, ScalarVar, IntArrayVar, OptVar, TensorListVar, expandable_to, broadcastable, same_shape

ENTRY = '_ZN2at6native5whereERKNS_6TensorE (aigen)'
IN_LOOP = False
LOOP_SAFE = True
_UNRESOLVED = frozenset([])
def get_unresolved_vars(): return _UNRESOLVED

_PARAMS = [('condition', 'Tensor')]
def get_params(): return _PARAMS

def get_constraints():
    condition = TensorVar('condition')
    # C++: TensorCompare.cpp:675 — where(condition) { return condition.nonzero_numpy(); }
    # -> nonzero_cpu: TensorAdvancedIndexing.cpp:2901/2937
    #    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(ComplexHalf, Half, BFloat16, Bool)
    # i.e. everything EXCEPT Float8_* (23,24,25,26,44) and wide uints (UInt16/32/64).
    valid = And(
        Not(Or(condition.dtype == 23, condition.dtype == 24,
                condition.dtype == 25, condition.dtype == 26,
                condition.dtype == 44)),
        Not(Or(condition.dtype == 27, condition.dtype == 28, condition.dtype == 29)),
    )
    return [Not(valid)]
