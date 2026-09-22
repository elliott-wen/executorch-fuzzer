#!/usr/bin/env python3
"""Replay exact corpus job(s) on the NXP Neutron NSYS simulator via the dedicated broker (15574)
and print eager vs device. Needs the nxp broker (job-port 15574) + nxp_client worker(s) up.
Usage: python _replay.py <job_id> ..."""
import os, sys, json, subprocess
CORPUS="/data/jwen929/mobile/corpus_v3/nxp"; PY="/data/jwen929/mobile/.venv/bin/python"
def path(j): s,n=j.split(":"); return f"{CORPUS}/{s}/{s}_{n}.job"
def replay(jids):
    env=dict(os.environ,PYTHONPATH="/data/jwen929",CUDA_VISIBLE_DEVICES="")
    cmd=[PY,"-m","mobile","feed","--host","127.0.0.1","--job-port","15574","--ctrl-port","15576",
         "--corpus",CORPUS,"--timeout","120"]+[path(j) for j in jids]
    p=subprocess.run(cmd,cwd="/data/jwen929",env=env,capture_output=True,text=True)
    for line in p.stdout.splitlines():
        line=line.strip()
        if not line.startswith("{"): continue
        r=json.loads(line); print(f"\n{r['job_id']} {r['op_chain']} -> {r['status']}  {r.get('detail','')[:80]}")
        for o in r.get("outputs",[]):
            print(f"  dtype={o['dtype']} shape={o['shape']}")
            print("  eager :",o.get("eager",[])[:10]); print("  device:",o.get("et",[])[:10])
if __name__=="__main__": replay(sys.argv[1:])
