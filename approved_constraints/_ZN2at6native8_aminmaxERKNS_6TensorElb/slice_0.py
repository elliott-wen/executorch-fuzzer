import sys
sys.path.insert(0, '/data/jwen929/pytorch/z3')
from z3 import *
from model import TensorVar, ScalarVar, IntArrayVar, OptVar, TensorListVar, expandable_to, broadcastable, same_shape
ENTRY = '_ZN2at6native8_aminmaxERKNS_6TensorElb (aigen)'
IN_LOOP = False
LOOP_SAFE = True
_UNRESOLVED = frozenset([])
def get_unresolved_vars() -> frozenset:
    return _UNRESOLVED
_PARAMS = [('self', 'Tensor'), ('dim', 'int'), ('keepdim', 'bool')]
def get_params() -> list:
    return _PARAMS
def get_constraints() -> list:
    self = TensorVar('self')
    dim = Int('dim')
    # _aminmax.dim (TensorCompare.cpp:821) delegates to at::aminmax
    # (ReduceOps.cpp:357, TORCH_META_FUNC(aminmax)).
    valid = And(
        # C++: aminmax_stub AT_DISPATCH_ALL_TYPES_AND3(Bool,Half,BFloat16):
        # {u8,i8,i16,i32,i64,f32,f64} + Bool + Half + BFloat16. No complex, no
        # barebones unsigned (UInt16/32/64), no Float8. probe: ComplexFloat/
        # UInt64/Float8_e8m0fnu -> "aminmax_cpu not implemented for '...'".
        Or(*(self.dtype == d for d in (0, 1, 2, 3, 4, 5, 6, 7, 11, 15))),
        # C++: ReduceOps.cpp:361 maybe_wrap_dim(dim, ndim); probe: dim=5 on a 2-D
        # tensor -> IndexError "Dimension out of range". 0-dim accepts dim in {0,-1}.
        Or(
            And(self.ndim == 0, Or(dim == 0, dim == -1)),
            And(self.ndim > 0, dim >= -self.ndim, dim < self.ndim),
        ),
        # C++: ReduceOps.cpp:362 zero_numel_check_dims -> ReduceOpsUtils.h:283
        # TORCH_CHECK_INDEX(self.size(dim) != 0); probe: reduce a 0-size dim ->
        # IndexError "Expected reduction dim N to have non-zero size".
        Implies(self.ndim > 0, self.size(dim) != 0),
    )
    return [Not(valid)]
