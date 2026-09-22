#!/usr/bin/env python3
"""Faithful cortex-m device repro / reason-capture (analysis_single.md steps 4-6).

Runs the EXACT stored artifacts from the corpus .job — the pre-lowered .pte, the same input
tensors, and the stored QUANTIZED reference the feeder diffs against — on the Corstone FVP via
fvp_runner.sh into a persistent out dir (uart.log survives). Then diffs device-vs-reference with
net/compare (the feeder's oracle), dropping ExecuTorch's prepended input-mutation outputs.

  - SKIP/CRASH: prints fvp_runner rc + the device uart.log Check-failed / abort lines (verbatim).
  - RAN: prints per-output compare verdict + worst eager-vs-device element (WRONG-VALUE evidence).

Usage: cm_repro.py <job_id> [--keep]
"""
import sys, os, subprocess, tempfile, shutil, json
from pathlib import Path

sys.path.insert(0, "/data/jwen929")
import torch
from mobile.executor import protocol as P
from mobile.executor import corpus as C
from mobile.executor import compare as CMP

MOBILE = Path("/data/jwen929/mobile")
FVP_RUNNER = MOBILE / "fvp_client" / "fvp_runner.sh"
CORPUS = MOBILE / "corpus_v3" / "cortex-m"

def load_stored(jid):
    frames = C.read_job(C.job_file(str(CORPUS), jid, "job"))
    _, pte, inputs = P.decode_job(P.job_frames_from_pushjob(frames))
    eager, user_pos = P.eager_from_pushjob(frames)
    _, desc = P.peek_jobinfo(frames)
    return pte, inputs, eager, user_pos, desc

def run(jid, keep=False):
    pte, inputs, eager, user_pos, desc = load_stored(jid)
    work = Path(tempfile.mkdtemp(prefix=f"cmrepro_{jid.replace(':','_')}_"))
    (work / "model.pte").write_bytes(pte)
    cmd = [str(FVP_RUNNER), "--backend", "cortex-m", "--pte", str(work / "model.pte"),
           "--out", str(work), "--target", "cortex-m55"]
    for i, t in enumerate(inputs):
        _, raw = P.tensor_to_meta_blob(t.contiguous())
        (work / f"in_{i}.bin").write_bytes(raw)
        cmd += ["--input", str(work / f"in_{i}.bin")]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc, proc = -9, None
    uart = (work / "uart.log").read_text(errors="replace") if (work / "uart.log").exists() else ""
    status = {-9: "TIMEOUT", 2: "SKIP", 0: "RAN"}.get(rc, "CRASH")
    out = {"jid": jid, "rc": rc, "status": status, "desc": desc, "inputs": inputs,
           "n_inputs": len(inputs), "work": str(work)}
    out["uart_err"] = [l.strip() for l in uart.splitlines()
                       if any(k in l for k in ("Check failed", "Error", "error", "abort",
                            "Fatal", "not supported", "not implemented", "Missing", "assert",
                            "Exception", "Fault", "status ==")) ][-8:]
    out["stderr_tail"] = ((proc.stderr or proc.stdout or "")[-400:]) if proc else ""
    if status == "RAN":
        outs = _rebuild(work, out_metas_from(jid))
        out["cmp"] = _diff(eager, outs, user_pos)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return out

def out_metas_from(jid):
    frames = C.read_job(C.job_file(str(CORPUS), jid, "job"))
    lean = P.job_frames_from_pushjob(frames)
    metas, _ = P.job_output_info(lean)
    return metas

def _rebuild(work, out_metas):
    files = sorted(work.glob("out_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
    outs = []
    for p in files:
        k = int(p.stem.split("_")[1])
        if k < len(out_metas):
            outs.append(P.tensor_from_meta_blob(out_metas[k], p.read_bytes()))
    return outs

def _diff(eager, dev, user_pos):
    if not dev:
        return {"verdict": "NO-OUTPUT"}
    et = dev
    if len(et) > len(eager):
        et = et[len(et) - len(eager):]
    try:
        status, detail = CMP.compare(eager, et)
    except Exception as e:
        status, detail = "CMPERR", str(e)
    worst = None
    for i, (e, d) in enumerate(zip(eager, et)):
        try:
            ef, df = e.float(), d.float()
            delta = (ef - df).abs()
            md = float(delta.max()) if delta.numel() else 0.0
            samp = None
            if delta.numel():
                idx = int(delta.argmax())
                samp = (float(ef.flatten()[idx]), float(df.flatten()[idx]))
            if worst is None or md > worst[1]:
                worst = (i, md, samp, tuple(e.shape), str(e.dtype), tuple(d.shape), str(d.dtype))
        except Exception as ex:
            worst = (i, -1, str(ex), None, str(e.dtype), None, str(d.dtype))
    return {"status": status, "detail": detail, "worst": worst}

if __name__ == "__main__":
    keep = "--keep" in sys.argv
    for jid in [a for a in sys.argv[1:] if not a.startswith("--")]:
        r = run(jid, keep=keep)
        print(f"\n=== {jid}  [{r['status']}] rc={r['rc']}  {r['desc']}")
        if r["status"] == "RAN" and r.get("cmp"):
            c = r["cmp"]; w = c.get("worst")
            print(f"  compare: {c.get('status')} — {c.get('detail')}")
            if w:
                i, md, samp, eshp, edt, dshp, ddt = w
                print(f"  worst out[{i}] max|delta|={md:.4g}  eager{eshp}/{edt} vs dev{dshp}/{ddt}")
                if samp:
                    print(f"    worst elem: eager={samp[0]}  device={samp[1]}")
        elif r.get("uart_err"):
            print("  uart device error(s):")
            for l in r["uart_err"]:
                print("    " + l[:160])
        else:
            print("  (no catchable device message) stderr tail: " + r.get("stderr_tail","").replace(chr(10)," | ")[:240])
