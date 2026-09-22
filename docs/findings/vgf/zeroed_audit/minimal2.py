import os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from minimal import probe, mk, A
F32="L0 = torch.randn((2,4), dtype=torch.float32)"
TWO="L0 = torch.randn((2,4), dtype=torch.float32)\nL1 = torch.randn((2,4), dtype=torch.float32)"
POS="L0 = torch.rand((2,4), dtype=torch.float32) + 0.5"
C=[
 ("ctrl: dup relu (known bug)",   mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.relu.default(L0)"],"n0, n1",F32)),
 ("3x dup relu",                  mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.relu.default(L0)",f"n2 = {A}.relu.default(L0)"],"n0, n1, n2",F32)),
 ("dup relu + distinct neg",      mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.relu.default(L0)",f"n2 = {A}.neg.default(L0)"],"n0, n1, n2",F32)),
 ("distinct neg + dup relu",      mk([f"n2 = {A}.neg.default(L0)",f"n0 = {A}.relu.default(L0)",f"n1 = {A}.relu.default(L0)"],"n2, n0, n1",F32)),
 ("dup relu on DIFFERENT inputs", mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.relu.default(L1)"],"n0, n1",TWO,"L0, L1")),
 ("same VALUE diff op (pos in: relu/abs)", mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.abs.default(L0)"],"n0, n1",POS)),
 ("same VALUE diff op chained (relu,abs(n0))", mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.abs.default(n0)"],"n0, n1",POS)),
 ("dup via mul1 / clone",         mk([f"n0 = {A}.mul.Scalar(L0, 1.0)",f"n1 = {A}.clone.default(L0)"],"n0, n1",F32)),
 ("w111 reduced: n0, lift_fresh_copy(n0)", mk([f"n0 = {A}.remainder.Tensor(L0, L1)",f"n1 = {A}.lift_fresh_copy.default(n0)"],"n0, n1",TWO,"L0, L1")),
 ("w111 shape: min(n0), log10(lfc(n0))", mk([f"n0 = {A}.remainder.Tensor(L0, L1)",f"n1 = {A}.min.default(n0)",f"n2 = {A}.lift_fresh_copy.default(n0)",f"n3 = {A}.log10.default(n2)"],"n1, n3",TWO,"L0, L1")),
 ("w108 reduced: fmod(n0,6), split(n0)", mk([f"n0 = {A}.index.Tensor(L0, [torch.tensor([0, 0], dtype=torch.int32)])",f"n1 = {A}.fmod.Scalar(n0, 6)",f"n2 = _first({A}.split_with_sizes_copy.default(n0, [2], 0))"],"n1, n2","L0 = torch.randn((2,), dtype=torch.float32)")),
 ("dup add-scalar",               mk([f"n0 = {A}.add.Scalar(L0, 1.0)",f"n1 = {A}.add.Scalar(L0, 1.0)"],"n0, n1",F32)),
 ("dup sigmoid",                  mk([f"n0 = {A}.sigmoid.default(L0)",f"n1 = {A}.sigmoid.default(L0)"],"n0, n1",F32)),
 ("dup relu, one consumed by neg too", mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.relu.default(L0)",f"n2 = {A}.neg.default(n1)"],"n0, n2",F32)),
 ("passthrough L0 + relu(L0)",    mk([f"n0 = {A}.relu.default(L0)"],"n0, L0",F32)),
]
for n,s in C: probe(n,s)
