import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from minimal import probe, mk, A
F32="L0 = torch.randn((2,4), dtype=torch.float32)"
TWO="L0 = torch.randn((2,4), dtype=torch.float32)\nL1 = torch.randn((2,4), dtype=torch.float32)"
C=[
 # 2 delegate outputs, each consumed by a PORTABLE op (fmod.Scalar / min.default), DISTINCT producers
 ("distinct producers -> portable ops",
   mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.abs.default(L0)",
       f"r0 = {A}.fmod.Scalar(n0, 6)",f"r1 = {A}.min.default(n1)"],"r0, r1",F32)),
 # same but DUPLICATE producers
 ("dup producers -> portable ops",
   mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.relu.default(L0)",
       f"r0 = {A}.fmod.Scalar(n0, 6)",f"r1 = {A}.min.default(n1)"],"r0, r1",F32)),
 # one delegate output feeds a portable op, the other is returned directly (w108 shape, distinct)
 ("distinct: portable(n0), n1 direct",
   mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.abs.default(L0)",f"r0 = {A}.fmod.Scalar(n0, 6)"],"r0, n1",F32)),
 ("dup: portable(n0), n1 direct",
   mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.relu.default(L0)",f"r0 = {A}.fmod.Scalar(n0, 6)"],"r0, n1",F32)),
 # pass-through style: delegate emits n0 and a no-op-equivalent of n0 via a *different* op
 ("n0 and slice-covering-all(n0)",
   mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.slice_copy.Tensor(n0, 0, 0, 2)"],"n0, n1",F32)),
 ("n0 and expand_copy(n0) same shape",
   mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.expand_copy.default(n0, [2,4])"],"n0, n1",F32)),
 ("n0 and view_copy(n0) same numel",
   mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.view_copy.default(n0, [4,2])"],"n0, n1",F32)),
 ("n0 and transpose_copy(n0)",
   mk([f"n0 = {A}.relu.default(L0)",f"n1 = {A}.transpose_copy.int(n0, 0, 1)"],"n0, n1",F32)),
 # w111 shape reduced: two delegate outputs = n0 and lift_fresh_copy(n0), each to a portable op
 ("n0->portable min ; lfc(n0)->portable log10",
   mk([f"n0 = {A}.remainder.Tensor(L0, L1)",f"n2 = {A}.lift_fresh_copy.default(n0)",
       f"r0 = {A}.min.default(n0)",f"r1 = {A}.log10.default(n2)"],"r0, r1",TWO,"L0, L1")),
 # single-op ancestor sanity: does aten.fill alone zero?  (COMP confound)
 ("fill alone", mk([f"n0 = {A}.fill.Scalar(L0, 3.5)"],"n0",F32)),
 ("full_like alone", mk([f"n0 = {A}.full_like.default(L0, 3.5)"],"n0",F32)),
]
for n,s in C: probe(n,s)
