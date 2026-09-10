#!/usr/bin/env python3
"""Host-only audit of the 7 non-SCALE ethos-u graphopt_mechanism rows.
Rebuilds the EXACT A/B graph deep_graphopt.py used ([target, breaking_sibling])
and reports: eager shapes/dtypes, user_pos, n_plan_outputs, and the eager
reference values -- so we can test (a) whether the harness output alignment is
sound, (b) whether the archived dev values could belong to the target at all."""
import os, sys, re, warnings, logging, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
os.environ.setdefault("MOBILE_BACKENDS","ethos-u")
sys.path.insert(0,"/data/jwen929")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from mobile.gen.export import build_job
from mobile.net import corpus as C
from executorch.exir._serialize._program import deserialize_pte_binary
CO="/data/jwen929/mobile/corpus_v2/ethos-u"

def parse(job):
    text=open(C.job_file(CO,job,"py")).read(); lines=text.splitlines()
    gi=next(i for i,l in enumerate(lines) if l.startswith("def g("))
    PRE="\n".join(lines[:gi+1]); nl={}; order=[]; ins={}; ret=[]; POST=""
    for l in lines[gi+1:]:
        m=re.match(r"\s+(n\d+)\s*=\s*(.+)",l)
        if m: nl[m.group(1)]=l.strip(); order.append(m.group(1)); ins[m.group(1)]=set(re.findall(r"\bn\d+\b",m.group(2)))
        r=re.match(r"\s+return\s*\((.+?),?\)\s*$",l)
        if r: ret=[x.strip() for x in r.group(1).split(",") if x.strip()]
        if l.startswith("LEAVES"): POST=l
    return PRE,nl,order,ins,ret,POST
def cone(outs,ins):
    seen=set(); st=list(outs)
    while st:
        x=st.pop()
        if x in seen or x not in ins: continue
        seen.add(x); st+=list(ins[x])
    return seen
def mksrc(PRE,nl,order,ins,POST,outs):
    keep=cone(outs,ins); body="\n".join("    "+nl[n] for n in order if n in keep)
    return PRE+"\n"+body+"\n    return ("+", ".join(outs)+",)\n\n"+POST+"\n"

def plan_out_shapes(pte):
    prog=deserialize_pte_binary(pte); prog=getattr(prog,"program",prog)
    plan=prog.execution_plan[0]
    out=[]
    for i in plan.outputs:
        v=plan.values[i].val
        out.append(tuple(v.sizes) if type(v).__name__=="Tensor" else type(v).__name__)
    return out

CASES=[("w78:202",4,"n6"),("w87:268",2,"n7"),("w59:999",3,"n2"),("w7:1024",1,"n5"),
       ("w61:77",0,"n3"),("w25:933",2,"n7"),("w66:845",1,"n1")]
for job,ti,sib in CASES:
    print(f"\n########## {job} out{ti} sibling={sib}")
    PRE,nl,order,ins,ret,POST=parse(job)
    tgt=ret[ti]
    print(f"  full return tuple = {ret}   target={tgt}  {nl.get(tgt,'?')}")
    print(f"  sibling {sib} = {nl.get(sib,'?')}")
    for label,outs in [("A[target]",[tgt]),("B[target,sib]",[tgt,sib])]:
        src=mksrc(PRE,nl,order,ins,POST,outs)
        for q in (True,):
            try: j=build_job(src,"ethos-u",q)
            except Exception as e:
                print(f"  {label} q={q}: BUILD_EXC {type(e).__name__}: {str(e)[:100]}"); continue
            if j.status!="READY":
                print(f"  {label} q={q}: {j.status} {j.detail[:90]}"); continue
            try: ps=plan_out_shapes(j.pte)
            except Exception as e: ps=f"EXC {type(e).__name__}"
            print(f"  {label} q={q}: READY deleg={getattr(j,'delegated_ops',None)} "
                  f"user_pos={j.user_pos} n_eager={len(j.eager)} plan_outs={ps}")
            for k,t in enumerate(j.eager):
                fl=[round(float(x),6) for x in t.flatten()[:6].tolist()]
                print(f"        eager[{k}] {tuple(t.shape)} {t.dtype} {fl}")
