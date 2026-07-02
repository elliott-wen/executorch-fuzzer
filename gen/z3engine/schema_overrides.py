
_OP_SCHEMAS = {
    "bitwise_and.Scalar": [("self", "Scalar"), ("other", "Tensor")],
    "bitwise_and.Scalar_Tensor": [("self", "Scalar"), ("other", "Tensor")],
    "bitwise_or_.Scalar": [("self", "Tensor"), ("other", "Scalar")],
    "bitwise_xor.Scalar": [("self", "Scalar"), ("other", "Tensor")],
    "bitwise_xor.Scalar_Tensor": [("self", "Scalar"), ("other", "Tensor")],
    "complex.out": [("real", "Tensor"), ("imag", "Tensor"), ("out", "Tensor")],
    "deg2rad.out": [("self", "Tensor"), ("out", "Tensor")],
    "log_normal_": [("self", "Tensor"), ("mean", "float"), ("std", "float"), ("generator", "Generator?")],
    "rad2deg_out": [("self", "Tensor"), ("out", "Tensor")],
    "unfold_copy": [("self", "Tensor"), ("dimension", "int"), ("size", "int"), ("step", "int")],
    "add_relu_out": [("self", "Tensor"), ("other", "Tensor"), ("alpha", "Scalar"), ("out", "Tensor")],
    "bitwise_and_": [("self", "Tensor"), ("other", "Scalar")],
    "bitwise_and_.Scalar": [("self", "Tensor"), ("other", "Scalar")],
    "bitwise_xor_": [("self", "Tensor"), ("other", "Scalar")],
    "bitwise_xor_.Scalar": [("self", "Tensor"), ("other", "Scalar")],
    "hann_window": [("window_length", "int"), ("dtype", "int?"), ("layout", "int?"), ("device", "int?"), ("pin_memory", "bool?")],
    "hann_window.periodic": [("window_length", "int"), ("periodic", "bool"), ("dtype", "int?"), ("layout", "int?"), ("device", "int?"), ("pin_memory", "bool?")],
    "__ilshift__.Scalar": [("self", "Tensor"), ("other", "Scalar")],
    "__ilshift__.Tensor": [("self", "Tensor"), ("other", "Tensor")],
    "__irshift__.Scalar": [("self", "Tensor"), ("other", "Scalar")],
    "__irshift__.Tensor": [("self", "Tensor"), ("other", "Tensor")],
    "__lshift__.Scalar": [("self", "Tensor"), ("other", "Scalar")],
    "__lshift__.Tensor": [("self", "Tensor"), ("other", "Tensor")],
    "__rshift__.Scalar": [("self", "Tensor"), ("other", "Scalar")],
    "__rshift__.Tensor": [("self", "Tensor"), ("other", "Tensor")],
    "normal.Tensor_float_out": [("self", "Tensor"), ("mean", "float"), ("std", "float"), ("generator", "Generator?"), ("out", "Tensor")],
}

def jit_to_model(type_str):
    return type_str

def schema_for_op(op_name):
    return object()

def named_params_from_schema(op_name):
    return _OP_SCHEMAS.get(op_name)
