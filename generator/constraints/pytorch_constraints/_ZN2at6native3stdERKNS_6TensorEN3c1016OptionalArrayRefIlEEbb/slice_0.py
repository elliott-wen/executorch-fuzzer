from z3 import *
from mobile.generator.constraints.model import TensorVar, ScalarVar, IntArrayVar, OptVar, TensorListVar, expandable_to, broadcastable, same_shape

ENTRY = '_ZN2at6native3stdERKNS_6TensorEN3c1016OptionalArrayRefIlEEbb (aigen)'
IN_LOOP = False
LOOP_SAFE = True
_UNRESOLVED = frozenset([])


def get_unresolved_vars():
    return _UNRESOLVED


_PARAMS = [('self', 'Tensor'), ('dim', 'int[]?'), ('unbiased', 'bool'), ('keepdim', 'bool')]


def get_params():
    return _PARAMS


def get_constraints():
    # at::native::std(Tensor self, OptionalIntArrayRef dim, bool unbiased, bool keepdim)
    # C++: ReduceOps.cpp:2110-2113 std(self, dim, unbiased, keepdim) ->
    #      at::std(self, dim, correction=unbiased?1:0, keepdim) -> std_var_out.
    # Preconditions:
    # (1) dtype: ReduceOps.cpp:1893-1894 TORCH_CHECK(isFloatingType ||
    #     isComplexType, "std and var only support floating point and complex
    #     dtypes"). Float8 excluded at runtime (std_var_all_cpu /
    #     AT_DISPATCH_FLOATING_TYPES, ReduceOps.cpp:1825-1828). Valid:
    #     Half=5, Float=6, Double=7, BFloat16=15, CHalf=8, CFloat=9, CDouble=10.
    # (2) dim range: make_reduction (ReduceOps.cpp:1932) -> dim_list_to_bitset ->
    #     maybe_wrap_dim on each provided dim. Each dim must lie in
    #     [-rank, rank) where rank = max(ndim, 1) (0-dim scalar accepts dim 0/-1).
    #     Out-of-range -> clean IndexError "Dimension out of range" (excluded).
    # AUDIT (torch probe): float/bf16/complex PASS; int/bool/float8 RuntimeError;
    #   dim within [-ndim,ndim) PASS, dim 2 / -3 on a 2-D tensor -> IndexError;
    #   0-dim scalar with dim 0 or -1 PASS.
    # `unbiased`, `keepdim` (bool) unconstrained.
    _self = TensorVar('self')
    _dim = OptVar('dim', IntArrayVar('dim.val'))
    _dt = _self.dtype
    valid_dtype = Or(_dt == 5, _dt == 6, _dt == 7, _dt == 15,
                     _dt == 8, _dt == 9, _dt == 10)

    rank = If(_self.ndim <= 0, IntVal(1), _self.ndim)
    arr = _dim.value
    dim_ok = And(*[
        Implies(
            And(_dim.present, IntVal(i) < arr.length),
            And(arr[i] >= -rank, arr[i] < rank),
        )
        for i in range(6)  # MAX_INT_ARRAY_LEN
    ])

    valid = And(valid_dtype, dim_ok)
    return [Not(valid)]


def get_axioms():
    return []
