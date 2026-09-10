from z3 import *
from mobile.generator.constraints.model import TensorVar, ScalarVar, IntArrayVar, OptVar, TensorListVar, expandable_to, broadcastable, same_shape, MAX_TENS
ENTRY = '_ZN2at6native28quantize_per_tensor_list_cpuEN3c108ArrayRefINS_6TensorEEERKS3_S6_NS1_10ScalarTypeE (aigen)'
IN_LOOP = False
LOOP_SAFE = True
_UNRESOLVED = frozenset([])
def get_unresolved_vars() -> frozenset:
    return _UNRESOLVED
_PARAMS = [('tensors', 'Tensor[]'), ('scales', 'Tensor'), ('zero_points', 'Tensor'), ('dtype', 'int')]
def get_params() -> list:
    return _PARAMS
def get_constraints() -> list:
    tensors = TensorListVar('tensors')
    scales = TensorVar('scales')
    zero_points = TensorVar('zero_points')
    dtype = Int('dtype')

    # C++: QTensor.cpp:75-89 quantize_per_tensor_list_cpu — for each i:
    #   at::quantize_per_tensor(tensors[i], scales[i].item<double>(),
    #                           zero_points[i].item<int64_t>(), dtype)
    # Each tensors[i] flows into PerTensorAffineQuantizer::quantize, which requires
    # scalar_type()==kFloat (Quantizer.cpp:163-166). probe: int element -> RuntimeError.
    per_tensor = []
    for i in range(MAX_TENS):
        ti = tensors.tensors[i]
        per_tensor.append(
            Implies(IntVal(i) < tensors.length, ti.dtype == 6))  # Float==6

    valid = And(
        And(*per_tensor),
        # scales[i] / zero_points[i] index dim 0 then call .item() (QTensor.cpp:84-85):
        # the indexed sub-tensor must be 0-d => scales/zero_points must be exactly 1-D
        # (multi-dim -> scales[i] has >1 element -> "cannot be converted to Scalar";
        #  0-d -> "select() cannot be applied to a 0-dim tensor"; short -> IndexError),
        # with dim-0 size covering the list length.
        scales.ndim == 1,
        zero_points.ndim == 1,
        scales.size(0) >= tensors.length,
        zero_points.size(0) >= tensors.length,
        # scales[i].item<double>() / zero_points[i].item<int64_t>() are read per
        # element. Pin both to Float (else a free complex/bool fill bypasses the value
        # clamp -> stray inf/huge -> "cannot be converted ... overflow") and bound
        # their values to a finite, always-valid window: scale in [0,1]; zero_point in
        # [0,127] (intersection over quint8[0,255]/qint8[-128,127]/qint32, the
        # Affine quantizer qmin/qmax check — probe zp=-1/256/-129/1e300 -> RuntimeError).
        scales.dtype == 6, zero_points.dtype == 6,
        scales.data_lo == 0, scales.data_hi == 1,
        zero_points.data_lo == 0, zero_points.data_hi == 127,
        # C++: Quantizer.cpp:141-145 new_qtensor — isQIntType(dtype). QInt non-sub-byte
        # {QInt8=12, QUInt8=13, QInt32=14}; sub-byte 16/17 excluded (concretizer cannot
        # build a torch.dtype for them). probe: Float dtype -> "not supported".
        Or(*(dtype == d for d in (12, 13, 14))),
    )
    return [Not(valid)]
