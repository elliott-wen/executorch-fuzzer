import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from lib import *
BR={}
with open("/data/jwen929/mobile/findings/vgf/bisect_results.tsv") as f:
    hdr=f.readline().rstrip("\n").split("\t"); ix={c:i for i,c in enumerate(hdr)}
    for line in f:
        r=line.rstrip("\n").split("\t")
        if len(r)<=ix["deleg_full"]: continue
        BR[(r[0],r[1])]=(r[ix["verdict"]],r[ix["needed_siblings"]],r[ix["needed_live"]])
for spec in sys.argv[1:]:
    job,oi=spec.split("#")
    v,ns,nl=BR[(job,oi)]
    needed=[x for x in (ns if v=="SIBLING_DEPENDENT" else nl).split(",") if x]
    print("="*70); print(job,oi,v,"needed=",needed)
    if v=="SIBLING_DEPENDENT":
        others,bos,target,ret=output_set_localizer(C.job_file(CO,job,"py"),int(oi))
        print("target=",target,"ret=",ret)
        print("--- A ---"); print("\n".join(l for l in bos([target]).splitlines() if l.startswith(("def g","    ","LEAVES","L"))))
        print("--- B ---"); print("\n".join(l for l in bos([target]+needed).splitlines() if l.startswith(("def g","    ","LEAVES","L"))))
    else:
        anc,bs,target=cone_localizer(C.job_file(CO,job,"py"),int(oi))
        print("target=",target)
        print("--- A ---"); print("\n".join(l for l in bs([]).splitlines() if l.startswith(("def g","    ","LEAVES","L"))))
        print("--- B ---"); print("\n".join(l for l in bs(needed).splitlines() if l.startswith(("def g","    ","LEAVES","L"))))
