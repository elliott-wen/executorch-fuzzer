from z3 import *
from mobile.generator.constraints.model import TensorVar, ScalarVar, IntArrayVar, OptVar, TensorListVar, expandable_to, broadcastable, same_shape

ENTRY = '_ZN2at6native5geqrfERKNS_6TensorE (aigen)'
IN_LOOP = False
LOOP_SAFE = True
_UNRESOLVED = frozenset([])
def get_unresolved_vars(): return _UNRESOLVED

_PARAMS = [('self', 'Tensor')]
def get_params(): return _PARAMS

def get_constraints():
    self = TensorVar('self')
    # C++: BatchLinearAlgebra.cpp:2379 — TORCH_CHECK(input.dim() >= 2)
    # C++: BatchLinearAlgebraKernel.cpp:449 — AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES
    #      -> {Float=6, Double=7, CFloat=9, CDouble=10}
    valid = And(
        self.ndim >= 2,
        Or(self.dtype == 6, self.dtype == 7, self.dtype == 9, self.dtype == 10),
    )
    return [Not(valid)]
