import sys
sys.path.insert(0, '/data/jwen929/pytorch/z3')
from z3 import *
from model import TensorVar, ScalarVar, IntArrayVar, OptVar, TensorListVar, expandable_to, broadcastable, same_shape
ENTRY = '_ZN2at6native34quantize_per_tensor_tensor_qparamsERKNS_6TensorES3_S3_N3c1010ScalarTypeE (aigen)'
IN_LOOP = False
LOOP_SAFE = True
_UNRESOLVED = frozenset([])
def get_unresolved_vars() -> frozenset:
    return _UNRESOLVED
_PARAMS = [('self', 'Tensor'), ('scale', 'Tensor'), ('zero_point', 'Tensor'), ('dtype', 'int')]
def get_params() -> list:
    return _PARAMS
def get_constraints() -> list:
    self = TensorVar('self')
    scale = TensorVar('scale')
    zero_point = TensorVar('zero_point')
    dtype = Int('dtype')
    valid = And(
        # C++: QTensor.cpp:66-73 quantize_per_tensor_tensor_qparams —
        #   make_per_tensor_affine_quantizer(scale.item().toDouble(),
        #                                    zero_point.item().toLong(), dtype)
        # .item() requires a single-element tensor (probe: 2-elem scale ->
        #   "a Tensor with 2 elements cannot be converted to Scalar").
        scale.numel_is_one(),
        zero_point.numel_is_one(),
        # C++: Quantizer.cpp:163-166 PerTensorAffineQuantizer::quantize —
        #   TORCH_CHECK(rtensor.scalar_type() == kFloat, "Quantize only works on
        #   Float Tensor"). probe: Double/Int self -> clean RuntimeError. Float==6.
        self.dtype == 6,
        # C++: Quantizer.cpp:141-145 new_qtensor —
        #   TORCH_CHECK(isQIntType(typeMetaToScalarType(dtype)), "... not supported")
        # QInt set: QInt8=12, QUInt8=13, QInt32=14, QUInt4x2=16, QUInt2x4=17.
        # Restrict to non-sub-byte {12,13,14}: the concretizer cannot materialize a
        # torch.dtype for sub-byte 16/17, so admitting them yields spurious failures.
        Or(*(dtype == d for d in (12, 13, 14))),
        # scale/zero_point are read via .item().toDouble()/.toLong(); pin both to a
        # real (Float) dtype so randomization + clamp is well-defined (a free
        # complex/bool fill bypasses the value clamp -> stray inf/huge -> overflow).
        scale.dtype == 6, zero_point.dtype == 6,
        # scale.item() must be finite and convertible to double (probe: finite scale
        # 0/neg/1e40 PASS; an unbounded random fill can produce values that overflow
        # the double conversion). Pin a finite, always-valid window.
        scale.data_lo == 0, scale.data_hi == 1,
        # zero_point.item() must lie within the target dtype's [qmin,qmax]
        # (Affine quantizer: probe zp=-1/256 quint8, -129 qint8, 1e300 ->
        #  "zero_point ... below/above bound" / "cannot be converted ... overflow").
        # [0,127] is the always-valid intersection over quint8[0,255], qint8[-128,127],
        # qint32[int32 range]; pinned via per-element data bounds (sound subset).
        zero_point.data_lo == 0, zero_point.data_hi == 127,
    )
    return [Not(valid)]
