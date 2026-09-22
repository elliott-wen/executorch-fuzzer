import sys,os,warnings,json
warnings.filterwarnings("ignore"); os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0,"/data/jwen929")
from mobile.gen.export import build_job
from mobile.executor.et_runner import run_pte
from mobile.executor import compare as cmp

HEAD='''import torch
torch.manual_seed(0)
L0 = torch.randn(4)
def g(L0):
'''
PROBES={
 # ---- section 2 probes (exact constructs from GRAPHOPT_REPORT.md) ----
 "s2_alone":                 "    n2 = torch.ops.aten.mul.Scalar(L0, 2.0)\n    a = torch.ops.aten.lift_fresh_copy.default(n2)\n    return (a,)\n",
 "s2_two_alias":             "    n2 = torch.ops.aten.mul.Scalar(L0, 2.0)\n    a = torch.ops.aten.lift_fresh_copy.default(n2)\n    b = torch.ops.aten.alias_copy.default(n2)\n    return (a, b)\n",
 "s2_self_plus_alias":       "    n2 = torch.ops.aten.mul.Scalar(L0, 2.0)\n    b = torch.ops.aten.alias_copy.default(n2)\n    return (n2, b)\n",
 "s2_leaf_two_alias":        "    a = torch.ops.aten.alias_copy.default(L0)\n    b = torch.ops.aten.alias_copy.default(L0)\n    return (a, b)\n",
 "s2_three_alias":           "    n2 = torch.ops.aten.mul.Scalar(L0, 2.0)\n    a = torch.ops.aten.lift_fresh_copy.default(n2)\n    b = torch.ops.aten.alias_copy.default(n2)\n    c = torch.ops.aten.view_copy.default(n2, [4])\n    return (a, b, c)\n",
 "s2_diff_tensors_CONTROL":  "    n2 = torch.ops.aten.mul.Scalar(L0, 2.0)\n    n3 = torch.ops.aten.add.Scalar(L0, 1.0)\n    a = torch.ops.aten.alias_copy.default(n2)\n    b = torch.ops.aten.alias_copy.default(n3)\n    return (a, b)\n",
 "s2_neutralized_plus0":     "    n2 = torch.ops.aten.mul.Scalar(L0, 2.0)\n    a = torch.ops.aten.lift_fresh_copy.default(n2)\n    b = torch.ops.aten.add.Scalar(torch.ops.aten.alias_copy.default(n2), 0.0)\n    return (a, b)\n",
 # ---- section 4 probes (view over unsupported vs supported source) ----
 "s4_unsupported_src":       "    n2 = torch.ops.aten.div.Scalar_mode(L0, -4, rounding_mode=None)\n    m = torch.ops.aten.max.default(n2)\n    a = torch.ops.aten.lift_fresh_copy.default(n2)\n    return (m, a)\n",
 "s4_supported_src_CONTROL": "    n2 = torch.ops.aten.mul.Scalar(L0, 2.0)\n    m = torch.ops.aten.max.default(n2)\n    a = torch.ops.aten.lift_fresh_copy.default(n2)\n    return (m, a)\n",
 # ---- section 1 reduced repro from w1:8 ----
 "s1_w1_8_reduced":          "    n0 = torch.ops.aten.cosh.out(L0, out=torch.empty((4,)))\n    n2 = torch.ops.aten.div.Scalar_mode(n0, -4, rounding_mode=None)\n    m = torch.ops.aten.max.default(n2)\n    a = torch.ops.aten.lift_fresh_copy.default(n2)\n    return (m, a)\n",
}
BK=os.environ["BK"]
res={}
for name,body in PROBES.items():
    src=HEAD+body+"LEAVES = [L0]\n"
    try:
        j=build_job(src,BK)
    except Exception as e:
        res[name]={"status":"BUILD_EXC","detail":str(e)[:150]}; continue
    if j.status!="READY":
        res[name]={"status":"BUILD:"+j.status,"detail":str(j.detail)[:150]}; continue
    try:
        raw=run_pte(j.pte,j.inputs)
    except Exception as e:
        res[name]={"status":"RUN_EXC","detail":str(e)[:150]}; continue
    et=cmp.select(raw,j.user_pos)
    if len(et)>len(j.eager): et=et[len(et)-len(j.eager):]
    outs=[]
    ok=True
    for i,(e,d) in enumerate(zip(j.eager,et)):
        v=cmp._cmp(e,d,0)[0]
        if v!="OK": ok=False
        el=e.flatten().tolist()[:6]; dl=d.flatten().tolist()[:6]
        zeroed = all(abs(x)<1e-9 for x in dl) and any(abs(x)>1e-6 for x in el)
        outs.append({"i":i,"verdict":v,"eager":[round(x,4) for x in el],"dev":[round(x,4) for x in dl],"ZEROED":zeroed})
    res[name]={"status":"RAN","all_ok":ok,"user_pos":list(j.user_pos) if j.user_pos else None,"outs":outs}
print(json.dumps({"backend":BK,"res":res},indent=1))
