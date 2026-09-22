import os,sys,warnings,logging,json,tempfile,glob,zipfile
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0,"/data/jwen929")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner
def blob(mod,args):
    ep=torch.export.export(mod,args)
    ee=to_edge_transform_and_lower(ep,partitioner=[CoreMLPartitioner()])
    gm=ee.exported_program().graph_module
    for n in gm.graph.nodes:
        if n.op=="get_attr": return bytes(getattr(gm,n.target).processed_bytes)
def show(label,mod,args):
    print("\n"+"="*70); print(label)
    b=blob(mod,args)
    # find end of JSON header
    depth=0;end=None
    for i,c in enumerate(b[:4096]):
        if c==0x7b: depth+=1
        elif c==0x7d:
            depth-=1
            if depth==0: end=i+1; break
    hdr=json.loads(b[:end].decode())
    print("  header:",json.dumps(hdr)[:400])
    rest=b[end:]
    d=tempfile.mkdtemp(); p=os.path.join(d,"m.mlpackage.zip"); open(p,"wb").write(rest)
    try:
        with zipfile.ZipFile(p) as z: z.extractall(d); print("  entries:",z.namelist()[:12])
    except Exception as e:
        print("  payload not zip:",e,"head:",rest[:32]); return
    for f in glob.glob(d+"/**/*.mil",recursive=True):
        print("  --- MIL ---"); print("   "+open(f).read().replace("\n","\n   ")[:2500])
class TwoAlias(torch.nn.Module):
    def forward(self,x):
        n2=torch.ops.aten.mul.Scalar(x,2.0)
        return (torch.ops.aten.lift_fresh_copy.default(n2), torch.ops.aten.alias_copy.default(n2))
class DiffCtl(torch.nn.Module):
    def forward(self,x):
        n2=torch.ops.aten.mul.Scalar(x,2.0); n3=torch.ops.aten.add.Scalar(x,1.0)
        return (torch.ops.aten.alias_copy.default(n2), torch.ops.aten.alias_copy.default(n3))
class S4(torch.nn.Module):
    def forward(self,x):
        n2=torch.ops.aten.div.Scalar_mode(x,-4,rounding_mode=None)
        return (torch.ops.aten.max.default(n2), torch.ops.aten.lift_fresh_copy.default(n2))
class S4ctl(torch.nn.Module):
    def forward(self,x):
        n2=torch.ops.aten.mul.Scalar(x,2.0)
        return (torch.ops.aten.max.default(n2), torch.ops.aten.lift_fresh_copy.default(n2))
class Neut(torch.nn.Module):
    def forward(self,x):
        n2=torch.ops.aten.mul.Scalar(x,2.0)
        return (torch.ops.aten.lift_fresh_copy.default(n2), torch.ops.aten.add.Scalar(torch.ops.aten.alias_copy.default(n2),0.0))
x=torch.randn(4)
for lbl,m in [("s2_two_alias",TwoAlias()),("s2_diff_tensors_CONTROL",DiffCtl()),("s4_unsupported_src",S4()),("s4_supported_src_CONTROL",S4ctl()),("s2_neutralized_plus0",Neut())]:
    show(lbl,m,(x,))
