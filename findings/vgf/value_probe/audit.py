#!/usr/bin/env python3
"""audit.py — re-verify VALUE / SCALE:* / ALIAS:* graph-opt rows on the real VGF runtime.

For each case: rebuild A (target isolated) and B (target + needed siblings/ancestors),
run both on mobile/vgf_client/vgf_runner.sh (A x2, B x NREP), then sub-classify the B-side
divergence with a corrected classifier that tests, in order:
  MISALIGN/SHAPE, NONFINITE(isfinite structure + inf-sign/nan), INT64_SENTINEL, FP16_SAT,
  ZEROED(all), PARTIAL_ZERO, CLOBBER(byte/near-equal to another live node tensor),
  PERMUTED/SHIFTED (same multiset, wrong positions), SCALE (guarded: >=8 ratios AND
  non-degenerate reference), NUMERIC (report max relative error).

Writes one TSV row per case.  Resumable: skips (job,out) already present in --out.
"""
import os, sys, json, subprocess, tempfile, pathlib, shutil, argparse, traceback, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/data/jwen929")
import warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import torch
from mobile.net import corpus as C, protocol as P, compare as cmp
from mobile.gen.export import build_job
from mobile.gen.diff.cone_predicate import cone_localizer, output_set_localizer, _load, _host_values

CO = "/data/jwen929/mobile/corpus_v4/vgf"
RUNNER_SH = "/data/jwen929/mobile/vgf_client/vgf_runner.sh"
NREP = int(os.environ.get("NREP", "3"))
FP16MAX = 65504.0
I64 = 9223372036854775807

COLS = ["job", "out", "verdict", "rec_mech", "tgt_op", "trigger_ops", "deleg_full",
        "A_deleg", "B_deleg", "A_status", "A_n_ok", "B_n_div", "B_n_rep", "repro",
        "actual_mech", "test_fired", "rel_err", "rel_scaled", "dtype", "shape", "misalign",
        "n_pte_out", "n_eager", "user_pos", "detail", "note"]


# ---------------------------------------------------------------- device plumbing
def run_dev(j, tag, keep=None):
    """Run one lowered job on the VGF runtime. Returns (list[Tensor] aligned to j.eager,
    n_pte_out, misalign_flag, err)."""
    d = pathlib.Path(tempfile.mkdtemp(prefix=f"va_{tag}_", dir=os.environ.get("VA_TMP", "/tmp")))
    try:
        pte = d / "m.pte"; pte.write_bytes(j.pte)
        cmd = [RUNNER_SH, "--pte", str(pte), "--out", str(d)]
        for i, t in enumerate(j.inputs):
            _, raw = P.tensor_to_meta_blob(t.contiguous())
            p = d / f"in_{i}.bin"; p.write_bytes(raw); cmd += ["--input", str(p)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            return None, 0, "", f"RUN_RC{r.returncode}:{(r.stderr or '')[-120:].replace(chr(10),' ')}"
        outs = sorted(d.glob("out_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
        n_pte = len(outs)
        up = j.user_pos
        mis = ""
        if up is not None and all(0 <= i < n_pte for i in up) and len(up) == len(j.eager):
            pick = [outs[i] for i in up]
        else:
            mis = f"UP_FALLBACK(up={up},n_pte={n_pte},n_eager={len(j.eager)})"
            pick = outs[n_pte - len(j.eager):] if n_pte >= len(j.eager) else outs
        ts = []
        for f, e in zip(pick, j.eager):
            buf = bytearray(f.read_bytes())
            t = torch.empty(0, dtype=e.dtype) if len(buf) == 0 else torch.frombuffer(buf, dtype=e.dtype)
            ts.append(t.reshape(e.shape) if t.numel() == e.numel() else t)
        return ts, n_pte, mis, ""
    except subprocess.TimeoutExpired:
        return None, 0, "", "RUN_TIMEOUT"
    except Exception as e:
        return None, 0, "", f"RUN_EXC:{type(e).__name__}:{e}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- classifier
def classify(ref, dev, node_vals, target):
    """-> (mech, test_fired, rel_err, rel_scaled, detail)"""
    if not isinstance(dev, torch.Tensor):
        return "NON_TENSOR", "t0", "", "", ""
    if ref.shape != dev.shape or ref.numel() != dev.numel():
        return "MISALIGN_OR_SHAPE", "t0", "", "", f"ref{tuple(ref.shape)} dev{tuple(dev.shape)}"
    r = ref.flatten().float(); d = dev.flatten().float()
    n = r.numel()

    # t1 non-finite structure (isfinite, not just isnan) + inf sign / nan-vs-inf
    rf, df = torch.isfinite(r), torch.isfinite(d)
    if bool((rf != df).any()):
        det = f"n_nonfin ref={int((~rf).sum())} dev={int((~df).sum())}"
        sat = bool(((d.abs() == FP16MAX) & (~rf)).any())
        if sat:
            return "FP16_SATURATION", "t1", "", "", det + " dev|.|==65504 where ref nonfinite"
        return "NONFINITE", "t1", "", "", det
    if bool((~rf).any()):
        bad = (~rf) & ((r.isnan() != d.isnan()) | (r != d))
        if bool(bad.any()):
            i = int(bad.nonzero()[0])
            return "NONFINITE", "t1b", "", "", f"pos{i} ref={float(r[i])} dev={float(d[i])}"

    # t2 int64 2^63 sentinel
    if ref.dtype in (torch.int64, torch.int32):
        rl = ref.flatten().to(torch.int64); dl = dev.flatten().to(torch.int64)
        if bool(((dl.abs() >= (1 << 62)) & (rl.abs() < (1 << 40))).any()):
            i = int(((dl.abs() >= (1 << 62)) & (rl.abs() < (1 << 40))).nonzero()[0])
            return "INT64_SENTINEL", "t2", "", "", f"pos{i} ref={int(rl[i])} dev={int(dl[i])}"

    # t3 fp16 saturation on finite values
    if ref.dtype in (torch.float32, torch.float64, torch.float16):
        satm = (d.abs() == FP16MAX) & (r.abs() > FP16MAX * 0.9)
        if bool(satm.any()) and bool((d.abs() == FP16MAX).sum()) >= 1 and bool((r.abs() != FP16MAX).all()):
            return "FP16_SATURATION", "t3", "", "", f"{int(satm.sum())}/{n} elems clamped to +-65504"

    # t4 all-zero
    if float(d.abs().max()) < 1e-6 and float(r.abs().max()) > 1e-3:
        return "ZEROED", "t4", "", "", ""

    # t5 partial zero: zeros where ref nonzero, rest ok
    zm = (d == 0) & (r.abs() > 1e-3)
    if bool(zm.any()):
        keep = ~zm
        rest_ok = (not bool(keep.any())) or bool(torch.allclose(d[keep], r[keep], rtol=1e-2, atol=1e-3))
        if rest_ok:
            return "PARTIAL_ZERO", "t5", "", "", f"{int(zm.sum())}/{n} elems zeroed, rest match"

    # t6 clobber / alias: equals another live tensor in the graph
    for nm, v in node_vals.items():
        if nm == target:
            continue
        vf = v.flatten()
        if vf.numel() != n:
            continue
        vff = vf.float()
        if v.dtype == dev.dtype and bool(torch.equal(vf, dev.flatten())):
            return f"CLOBBER_EXACT:{nm}", "t6", "", "", "byte-identical to another live tensor"
        if torch.isfinite(vff).all() and bool(torch.allclose(vff, d, rtol=1e-3, atol=1e-4)):
            return f"CLOBBER_NEAR:{nm}", "t6", "", "", "near-equal to another live tensor"

    # t7 permutation / shift: same multiset, wrong positions
    if n >= 2:
        rs, _ = torch.sort(r); ds, _ = torch.sort(d)
        if bool(torch.allclose(rs, ds, rtol=1e-3, atol=1e-4)) and not bool(torch.allclose(r, d, rtol=1e-3, atol=1e-4)):
            sh = ""
            for k in range(1, min(n, 8)):
                if bool(torch.allclose(d[:n - k], r[k:], rtol=1e-3, atol=1e-4)):
                    sh = f" shift+{k}"; break
                if bool(torch.allclose(d[k:], r[:n - k], rtol=1e-3, atol=1e-4)):
                    sh = f" shift-{k}"; break
            return ("SHIFTED" if sh else "PERMUTED"), "t7", "", "", f"same multiset, {int((r != d).sum())}/{n} positions differ{sh}"

    # near-permutation: a small number of positions hold values that exist elsewhere in ref
    if n >= 4:
        diff = (r - d).abs() > (1e-3 + 1e-2 * r.abs())
        nd = int(diff.sum())
        if 0 < nd <= max(2, n // 4):
            dv = d[diff]
            inref = torch.tensor([bool((((r - x).abs() <= 1e-3 + 1e-2 * r.abs()).any())) for x in dv])
            if bool(inref.all()):
                return "MISPLACED_ELEMS", "t7b", "", "", f"{nd}/{n} positions hold a value taken from elsewhere in ref"

    # t8 guarded SCALE
    m = r.abs() > 1e-3
    ratio_note = ""
    if bool(m.any()):
        ratio = d[m] / r[m]
        ratio = ratio[torch.isfinite(ratio)]
        nz = int(ratio.numel())
        ref_nondeg = float(r.std()) > 0.01 * max(1e-12, float(r.abs().mean()))
        if nz >= 2:
            rm, rsd = float(ratio.mean()), float(ratio.std()) if nz > 1 else 0.0
            consistent = rsd < 0.05 * max(1e-9, abs(rm)) and abs(rm - 1) > 0.05
            if consistent and nz >= 8 and ref_nondeg:
                return f"SCALE:{rm:.4g}", "t8", "", "", f"n_ratio={nz} std/mean={rsd/max(1e-12,abs(rm)):.3g} ref nondegenerate"
            if consistent:
                ratio_note = (f"ratio={rm:.4g} consistent but VACUOUS "
                              f"(n_ratio={nz}{'' if ref_nondeg else ', ref DEGENERATE'})")

    # t9 numeric drift
    fin = rf & df
    if bool(fin.any()):
        rel = float(((d[fin] - r[fin]).abs() / r[fin].abs().clamp_min(1e-12)).max())
        adiff = float((d[fin] - r[fin]).abs().max())
    else:
        rel, adiff = float("nan"), float("nan")
    scale = max(float(r[fin].abs().max()) if bool(fin.any()) else 0.0, 1e-12)
    rel_s = adiff / scale
    det = f"max|delta|={adiff:.4g} refmax={scale:.4g}"
    if ratio_note:
        det += "; " + ratio_note
    mech = "NUMERIC" if ref.dtype.is_floating_point or ref.dtype.is_complex else "INT_WRONG_VALUE"
    return mech, "t9", f"{rel:.4g}", f"{rel_s:.4g}", det


# ---------------------------------------------------------------- per case
def do_case(cs):
    job, out_idx, verdict, needed = cs["job"], cs["out"], cs["verdict"], cs["needed"]
    py = C.job_file(CO, job, "py")
    row = {c: "" for c in COLS}
    row.update({"job": job, "out": out_idx, "verdict": verdict, "rec_mech": cs["mech"],
                "tgt_op": cs["tgt_op"], "trigger_ops": cs["trigger_ops"],
                "deleg_full": cs["deleg_full"], "B_n_rep": NREP})
    pre, leaves, node_val, order, deps, ret = _load(py)
    ns = _host_values(pre, node_val, order)
    node_vals = {k: v for k, v in ns.items() if isinstance(v, torch.Tensor)}

    if verdict == "SIBLING_DEPENDENT":
        others, bos, target, ret2 = output_set_localizer(py, out_idx)
        srcA = bos([target]); posA = 0
        sel = [o for o in ret2 if o in set([target] + needed)]
        srcB = bos([target] + needed); posB = sel.index(target)
    else:
        anc, build_src, target = cone_localizer(py, out_idx)
        srcA = build_src([]); posA = -1
        srcB = build_src(needed); posB = -1
    row["note"] = f"target={target} needed={','.join(needed)}"

    # --- A side (must be CORRECT)
    jA = build_job(srcA, "vgf", False)
    if jA.status != "READY":
        row["A_status"] = f"BUILD:{jA.status}"; row["repro"] = "BUILD_FAIL"; return row
    row["A_deleg"] = jA.delegated_ops
    nok = 0; astat = []
    for k in range(2):
        ts, npte, mis, err = run_dev(jA, "A")
        if ts is None:
            astat.append(err); continue
        s, dt = cmp._cmp(jA.eager[posA], ts[posA], 0)
        astat.append(s)
        if s != "MISMATCH":
            nok += 1
    row["A_n_ok"] = nok
    row["A_status"] = ",".join(astat)

    # --- B side
    jB = build_job(srcB, "vgf", False)
    if jB.status != "READY":
        row["repro"] = "BUILD_FAIL"; row["detail"] = f"B BUILD:{jB.status}"; return row
    row["B_deleg"] = jB.delegated_ops
    eB = jB.eager[posB]
    row["dtype"] = str(eB.dtype).replace("torch.", ""); row["shape"] = str(tuple(eB.shape))
    row["n_eager"] = len(jB.eager); row["user_pos"] = str(jB.user_pos)
    ndiv = 0; first = None; runerr = ""
    for k in range(NREP):
        ts, npte, mis, err = run_dev(jB, "B")
        if ts is None:
            runerr = err; continue
        row["n_pte_out"] = npte
        if mis:
            row["misalign"] = mis
        s, dt = cmp._cmp(eB, ts[posB], 0)
        if s == "MISMATCH":
            ndiv += 1
            if first is None:
                first = ts[posB]
                row["detail"] = dt
    row["B_n_div"] = ndiv
    if first is None:
        row["repro"] = "B_NOREPRO" if not runerr else "RUN_FAIL"
        row["actual_mech"] = "" ; row["note"] += " " + runerr
        return row
    if nok == 0:
        row["repro"] = "A_ALSO_DIVERGES"
    elif nok < 2 or ndiv < NREP:
        row["repro"] = "FLAKY"
    else:
        row["repro"] = "AB_OK"
    dd = os.environ.get("VA_DUMP")
    if dd:
        os.makedirs(dd, exist_ok=True)
        torch.save({"ref": eB, "dev": first, "node_vals": node_vals, "target": target,
                    "srcA": srcA, "srcB": srcB, "posA": posA, "posB": posB,
                    "case": cs, "A_deleg": int(jA.delegated_ops), "B_deleg": int(jB.delegated_ops),
                    "eagerB": jB.eager, "inputsB": jB.inputs,
                    "A_n_ok": nok, "B_n_div": ndiv},
                   os.path.join(dd, f"{job.replace(':','_')}__{out_idx}.pt"))
    m, tf, rel, rels, det = classify(eB, first, node_vals, target)
    row["actual_mech"] = m; row["test_fired"] = tf; row["rel_err"] = rel; row["rel_scaled"] = rels
    row["detail"] = (row["detail"] + " | " + det).strip(" |")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    cases = json.load(open(a.cases))
    mine = [c for i, c in enumerate(cases) if i % a.nshards == a.shard]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            p = l.split("\t")
            if len(p) > 2 and p[0] != "job":
                done.add((p[0], p[1]))
    new = not os.path.exists(a.out)
    fh = open(a.out, "a", buffering=1)
    if new:
        fh.write("\t".join(COLS) + "\n")
    for cs in mine:
        if (cs["job"], str(cs["out"])) in done:
            continue
        t0 = time.time()
        try:
            row = do_case(cs)
        except Exception as e:
            row = {c: "" for c in COLS}
            row.update({"job": cs["job"], "out": cs["out"], "verdict": cs["verdict"],
                        "rec_mech": cs["mech"], "tgt_op": cs["tgt_op"], "repro": "EXC",
                        "detail": f"{type(e).__name__}:{e}"[:200]})
            traceback.print_exc()
        row["note"] = f"{row['note']} t={time.time()-t0:.0f}s"
        fh.write("\t".join(str(row[c]).replace("\t", " ").replace("\n", " ") for c in COLS) + "\n")
        print(f"[{a.shard}] {cs['job']}:{cs['out']} {cs['mech']} -> {row['repro']} {row['actual_mech']} rel={row['rel_err']} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
