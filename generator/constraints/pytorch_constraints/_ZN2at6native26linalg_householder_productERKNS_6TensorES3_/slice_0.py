from z3 import *
from mobile.generator.constraints.model import TensorVar, ScalarVar, IntArrayVar, OptVar, TensorListVar, expandable_to, broadcastable, same_shape, MAX_DIM
ENTRY = '_ZN2at6native26linalg_householder_productERKNS_6TensorES3_ (aigen)'
IN_LOOP = False
LOOP_SAFE = True
_UNRESOLVED = frozenset([])
def get_unresolved_vars() -> frozenset:
    return _UNRESOLVED
_PARAMS = [('input', 'Tensor'), ('tau', 'Tensor')]
def get_params() -> list:
    return _PARAMS
def get_constraints() -> list:
    inp = TensorVar('input')
    tau = TensorVar('tau')

    # Batch dims must match: input.shape[:-2] == tau.shape[:-1].
    # With input.dim()-tau.dim()==1, input.shape[:-2] has (input.ndim-2) dims and
    # tau.shape[:-1] has (tau.ndim-1) == (input.ndim-2) dims; pair them index-by-index.
    # C++: BatchLinearAlgebra.cpp:2593-2600 — actual_batch_tau_shape.equals(expected).
    batch_eq = []
    for k in range(MAX_DIM):
        # input batch dim k is input.sizes[k]; corresponding tau batch dim is tau.sizes[k]
        # (both batch ranges start at index 0). Active only when k < input.ndim - 2.
        batch_eq.append(
            Implies(IntVal(k) < inp.ndim - 2,
                    Select(inp.sizes, IntVal(k)) == Select(tau.sizes, IntVal(k))))

    valid = And(
        # C++: BatchLinearAlgebra.cpp:2579 — input must have >= 2 dimensions.
        inp.ndim >= 2,
        # C++: BatchLinearAlgebra.cpp:2580-2582 — input.size(-2) >= input.size(-1).
        inp.size(-2) >= inp.size(-1),
        # C++: BatchLinearAlgebra.cpp:2583-2585 — input.size(-1) >= tau.size(-1).
        inp.size(-1) >= tau.size(-1),
        # C++: BatchLinearAlgebra.cpp:2587-2592 — input.dim() - tau.dim() == 1.
        inp.ndim - tau.ndim == 1,
        # C++: BatchLinearAlgebra.cpp:2601-2607 — tau.scalar_type() == input.scalar_type().
        tau.dtype == inp.dtype,
        # Kernel "orgqr_cpu" is backed by LAPACK s/d/c/z orgqr (BatchLinearAlgebra.cpp
        # :323-324, :955-959), so only Float=6, Double=7, ComplexFloat=9,
        # ComplexDouble=10 are implemented. probe: Half/BFloat16/ComplexHalf/Int all
        # -> "orgqr_cpu not implemented for ...". (NOT the generic floating set.)
        Or(*(inp.dtype == d for d in (6, 7, 9, 10))),
        # batch-dim equality (BatchLinearAlgebra.cpp:2593-2600)
        And(*batch_eq),
    )
    return [Not(valid)]
