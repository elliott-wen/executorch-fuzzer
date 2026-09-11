"""nf_common.py — shared helpers for the NONFINITE audit.

Reconstructs the A/B graphs for a graphopt_all.tsv row exactly as tmp/vgf_go_deep.py does,
runs a lowered .pte on the VGF local runtime (vgf_client/vgf_runner.sh), and classifies.
"""
import os, sys, subprocess, tempfile, pathlib, ast
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/data/jwen929")
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from mobile.executor import corpus as C, protocol as P, compare as cmp
from mobile.gen.export import build_job
from mobile.gen.diff.cone_predicate import cone_localizer, output_set_localizer, _load, _host_values

ROOT = "/data/jwen929/mobile"
CO = f"{ROOT}/corpus_v4/vgf"
RUNNER_SH = f"{ROOT}/vgf_client/vgf_runner.sh"


def load_rows():
    """graphopt_all.tsv joined with bisect_results.tsv (for needed_siblings / needed_live)."""
    need = {}
    with open(f"{ROOT}/findings/vgf/bisect_results.tsv") as f:
        hdr = f.readline().rstrip("\n").split("\t"); ix = {c: i for i, c in enumerate(hdr)}
        for line in f:
            r = line.rstrip("\n").split("\t")
            if len(r) <= ix["needed_live"]:
                continue
            v = r[ix["verdict"]]
            s = r[ix["needed_siblings"]] if v == "SIBLING_DEPENDENT" else r[ix["needed_live"]]
            need[(r[0], r[1])] = [x for x in s.split(",") if x]
    rows = []
    # NOTE: graphopt_all.tsv has NO header row (984 data rows, all shards concatenated).
    GHDR = ["job","out","verdict","tgt_op","trigger_ops","ab","n_div","n_rep","mechanism"]
    with open(f"{ROOT}/findings/vgf/graphopt_all.tsv") as f:
        hdr = GHDR
        for line in f:
            r = line.rstrip("\n").split("\t")
            d = dict(zip(hdr, r))
            d["needed"] = need.get((d["job"], d["out"]), [])
            rows.append(d)
    return rows


def build_ab(job_id, out_idx, verdict):
    """-> (srcA, srcB, target, posA, posB, node_vals)  matching vgf_go_deep.py exactly."""
    py = C.job_file(CO, job_id, "py")
    pre, leaves, node_val, order, deps, ret = _load(py)
    ns = _host_values(pre, node_val, order)
    node_vals = {k: v for k, v in ns.items() if isinstance(v, torch.Tensor)}
    return py, ret, node_vals


def srcs_for(job_id, out_idx, verdict, needed):
    py = C.job_file(CO, job_id, "py")
    if verdict == "SIBLING_DEPENDENT":
        others, bos, target, ret = output_set_localizer(py, int(out_idx))
        srcA = bos([target]); posA = 0
        sel = [o for o in ret if o in set([target] + needed)]
        srcB = bos([target] + needed); posB = sel.index(target)
    else:
        anc, build_src, target = cone_localizer(py, int(out_idx))
        srcA = build_src([]); posA = -1
        srcB = build_src(needed); posB = -1
    return srcA, srcB, target, posA, posB


def leaf_tensors(job_id):
    """The graph's leaf input tensors, as actually materialised by the corpus .py (seeded)."""
    py = C.job_file(CO, job_id, "py")
    pre, leaves, node_val, order, deps, ret = _load(py)
    ns = {}
    exec(compile(ast.Module(body=pre, type_ignores=[]), "<pre>", "exec"), ns)
    return {n: ns[n] for n in leaves if isinstance(ns.get(n), torch.Tensor)}


def vgf_run(job, tag, timeout=900):
    """Run job.pte on the local VGF runtime -> list of raw output byte strings, or (None, err)."""
    d = pathlib.Path(tempfile.mkdtemp(prefix=f"nf_{tag}_"))
    (d / "m.pte").write_bytes(job.pte)
    cmd = [RUNNER_SH, "--pte", str(d / "m.pte"), "--out", str(d)]
    for i, t in enumerate(job.inputs):
        _, raw = P.tensor_to_meta_blob(t.contiguous())
        p = d / f"in_{i}.bin"; p.write_bytes(raw); cmd += ["--input", str(p)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    if r.returncode != 0:
        log = (d / "run.log")
        tail = log.read_text()[-300:].replace("\n", " | ") if log.exists() else r.stderr[-300:]
        return None, f"rc={r.returncode} {tail}"
    outs = sorted(d.glob("out_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
    return [p.read_bytes() for p in outs], None


def align(raws, job):
    """Turn raw device bytes into tensors aligned to job.eager, honouring user_pos + trimming."""
    eager = job.eager
    sel = cmp.select(raws, job.user_pos)
    if len(sel) > len(eager):
        sel = sel[len(sel) - len(eager):]
    if len(sel) != len(eager):
        return None
    out = []
    for e, b in zip(eager, sel):
        n = e.numel() * e.element_size()
        if len(b) != n:
            return None
        out.append(torch.frombuffer(bytearray(b), dtype=e.dtype).reshape(e.shape))
    return out


def nf_signature(t):
    """Compact non-finite signature of a tensor."""
    f = t.flatten().float()
    return dict(n=f.numel(), nan=int(f.isnan().sum()), pinf=int((f == float("inf")).sum()),
                ninf=int((f == float("-inf")).sum()),
                absmax_fin=(float(f[f.isfinite()].abs().max()) if bool(f.isfinite().any()) else 0.0))


def classify(ref, dev):
    """(status, mech_go_deep, nf_kind) — mech_go_deep replicates tmp/vgf_go_deep.py::mech()."""
    st, det = cmp._cmp(ref, dev, 0)
    r = ref.flatten().float(); d = dev.flatten().float()
    if r.numel() != d.numel():
        return st, "SHAPE", "shape"
    rf, df = torch.isfinite(r), torch.isfinite(d)
    mech = None
    if (rf != df).any():
        mech = "NONFINITE"
    elif d.abs().max() < 1e-6 and r.abs().max() > 1e-3:
        mech = "ZEROED"
    else:
        mech = "VALUE"
    # finer non-finite kind
    rs, ds = nf_signature(ref), nf_signature(dev)
    kinds = []
    if rs["nan"] + rs["pinf"] + rs["ninf"] == 0 and ds["nan"] + ds["pinf"] + ds["ninf"] == 0:
        kinds.append("both_finite")
    if rs["nan"] + rs["pinf"] + rs["ninf"] > 0:
        kinds.append("ref_nonfinite")
    if ds["nan"] + ds["pinf"] + ds["ninf"] > 0:
        kinds.append("dev_nonfinite")
    # fp16 saturation detection: device hits +/-65504 where ref is non-finite or huge
    sat = bool(((d.abs() == 65504.0)).any())
    if sat:
        kinds.append("fp16_sat")
    return st, mech, ",".join(kinds) or "?"
