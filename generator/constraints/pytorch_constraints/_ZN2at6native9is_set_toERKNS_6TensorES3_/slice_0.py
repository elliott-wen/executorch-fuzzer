from z3 import *
from mobile.generator.constraints.model import TensorVar, ScalarVar, IntArrayVar, OptVar, TensorListVar, expandable_to, broadcastable, same_shape

ENTRY = '_ZN2at6native9is_set_toERKNS_6TensorES3_ (aigen)'
IN_LOOP = False
LOOP_SAFE = True
_UNRESOLVED = frozenset([])
def get_unresolved_vars(): return _UNRESOLVED

_PARAMS = [('self', 'Tensor'), ('tensor', 'Tensor')]
def get_params(): return _PARAMS

def get_constraints():
    self = TensorVar('self')
    tensor = TensorVar('tensor')
    # C++: aten/src/ATen/native/TensorProperties.cpp:159 — is_set_to(self, src) only reads
    # storage()/storage_offset()/dim()/size()/stride() and returns a bool. No TORCH_CHECK,
    # no AT_DISPATCH, no kernel: probed over all dtypes/shapes -> never crashes. Fully
    # permissive; valid is trivially true so the failure region is empty.
    valid = BoolVal(True)
    return [Not(valid)]
