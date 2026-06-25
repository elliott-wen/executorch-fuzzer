import sys; sys.path.insert(0,'/data/jwen929/pytorch/z3')
from z3 import *
from model import TensorVar, ScalarVar, IntArrayVar, OptVar, TensorListVar, expandable_to, broadcastable, same_shape

ENTRY = '_ZN2at6native9hardtanh_ERNS_6TensorERKN3c106ScalarES6_ (aigen)'
IN_LOOP = False
LOOP_SAFE = True
_UNRESOLVED = frozenset([])
def get_unresolved_vars(): return _UNRESOLVED

_PARAMS = [('self', 'Tensor'), ('min_val', 'Scalar'), ('max_val', 'Scalar')]
def get_params(): return _PARAMS

def get_constraints():
    self = TensorVar('self')
    min_val = ScalarVar('min_val')
    max_val = ScalarVar('max_val')
    # hardtanh_ -> hardtanh_out -> at::clamp_out (boundaries not type-promoting).
    # C++: aten/src/ATen/native/Activation.cpp:442 — TORCH_CHECK(self.scalar_type()!=kBool)
    # C++: aten/src/ATen/native/cpu/TensorCompareKernel.cpp:359 — clamp_scalar dispatch is
    #      AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, ...): ALL_TYPES = u8,i8,i16,i32,i64,
    #      f32,f64 {0,1,2,3,4,6,7} plus Half(5) and BFloat16(15). Complex, Float8, Bool, and
    #      barebones UInt16/32/64 raise RuntimeError ("clamp not supported"/"not implemented").
    CLAMP_DTYPES = (0, 1, 2, 3, 4, 5, 6, 7, 15)
    valid = Or(*(self.dtype == d for d in CLAMP_DTYPES))
    # C++: Activation.cpp:450 — for kByte (uint8, dtype 0) the integral boundary
    # toLong() values must be non-negative, else "cannot do hardtanh on an
    # unsigned type with negative limits". Gate on the active scalar variant.
    def nonneg(s):
        # toLong() takes the real part for floating/complex scalars, the int for integral.
        return If(s.is_integral(include_bool=True), s.int_val >= IntVal(0), s.real_val >= RealVal(0))
    uint8_ok = Or(self.dtype != IntVal(0), And(nonneg(min_val), nonneg(max_val)))
    return [Not(And(valid, uint8_ok))]
