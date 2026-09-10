#!/usr/bin/env python3
"""reclass.py — offline corrected sub-classification of the dumped (ref, dev) pairs."""
import os, sys, glob, json, math
sys.path.insert(0, "/data/jwen929")
import warnings; warnings.filterwarnings("ignore")
import torch

W = "/tmp/claude-2496112/-data-jwen929-mobile/e07c5b38-7e6e-4875-af54-f33813aae47b/scratchpad/va"
DUMP = f"{W}/dump"
FP16MAX = 65504.0
BIG = float(1 << 48)          # 2.8e14: past int64/UB-shift garbage territory
P63 = 9223372036854775808.0

def cls(ref, dev, node_vals, target):
    if ref.shape != dev.shape:
        return "MISALIGN_OR_SHAPE", "t0", None, None, f"ref{tuple(ref.shape)} dev{tuple(dev.shape)}"
    r = ref.flatten().float(); d = dev.flatten().float(); n = r.numel()
    rf, df = torch.isfinite(r), torch.isfinite(d)

    # --- t1 non-finite structure (isfinite, catching infinities not just nan)
    if bool((rf != df).any()):
        det = f"nonfin ref={int((~rf).sum())} dev={int((~df).sum())}"
        if bool(((d.abs() == FP16MAX) & (~rf)).any()):
            return "NONFINITE_FP16SAT", "t1", None, None, det + "; dev=+-65504 where ref nonfinite"
        return "NONFINITE", "t1", None, None, det
    if bool((~rf).any()):
        bad = (~rf) & ((r.isnan() != d.isnan()) | (r != d))
        if bool(bad.any()):
            i = int(bad.nonzero()[0])
            return "NONFINITE", "t1b", None, None, f"pos{i} ref={float(r[i])} dev={float(d[i])}"

    # --- t2 extreme-magnitude / int64-overflow regime: comparison is meaningless
    rmax = float(r.abs().max()) if n else 0.0
    dmax = float(d.abs().max()) if n else 0.0
    if max(rmax, dmax) >= BIG:
        tag = ""
        if abs(max(rmax, dmax) - P63) / P63 < 1e-6: tag = " ==2^63"
        elif math.log2(max(rmax, dmax)) % 1.0 < 1e-6: tag = f" ==2^{round(math.log2(max(rmax,dmax)))}"
        return "EXTREME_MAGNITUDE", "t2", None, None, f"|ref|max={rmax:.4g} |dev|max={dmax:.4g}{tag}"

    # --- t3 fp16 saturation on finite refs
    if ref.dtype.is_floating_point:
        satm = (d.abs() == FP16MAX) & (r.abs() != FP16MAX)
        if bool(satm.any()):
            return "FP16_SATURATION", "t3", None, None, f"{int(satm.sum())}/{n} clamped to +-65504"

    # --- t4 all-zero device / all-zero reference
    if dmax < 1e-6 and rmax > 1e-3:
        return "ZEROED", "t4", None, None, f"|ref|max={rmax:.4g}"
    # --- t5 partial zero
    zm = (d == 0) & (r.abs() > 1e-3)
    if bool(zm.any()):
        keep = ~zm
        if (not bool(keep.any())) or bool(torch.allclose(d[keep], r[keep], rtol=1e-2, atol=1e-3)):
            return "PARTIAL_ZERO", "t5", None, None, f"{int(zm.sum())}/{n} zeroed, rest match"

    # --- t6 clobber (byte-identical / near-identical to another live tensor)
    for nm, v in node_vals.items():
        if nm == target: continue
        vf = v.flatten()
        if vf.numel() != n: continue
        if v.dtype == dev.dtype and bool(torch.equal(vf, dev.flatten())):
            return f"CLOBBER_EXACT:{nm}", "t6", None, None, "byte-identical to another live tensor"
        vff = vf.float()
        if bool(torch.isfinite(vff).all()) and bool(torch.allclose(vff, d, rtol=1e-3, atol=1e-4)):
            return f"CLOBBER_NEAR:{nm}", "t6", None, None, "near-equal to another live tensor"

    # --- t4b reference is all-zero but the device produced non-zero values (not a clobber)
    if rmax < 1e-6 and dmax > 1e-3:
        return "REF_ZERO_DEV_NONZERO", "t4b", None, None, f"ref all~0, |dev|max={dmax:.4g}"

    # --- t7 permutation / shift / misplacement
    if n >= 2:
        rs, _ = torch.sort(r); ds, _ = torch.sort(d)
        if bool(torch.allclose(rs, ds, rtol=1e-3, atol=1e-4)):
            sh = ""
            for k in range(1, min(n, 8)):
                if bool(torch.allclose(d[:n-k], r[k:], rtol=1e-3, atol=1e-4)): sh = f" shift+{k}"; break
                if bool(torch.allclose(d[k:], r[:n-k], rtol=1e-3, atol=1e-4)): sh = f" shift-{k}"; break
            return ("SHIFTED" if sh else "PERMUTED"), "t7", None, None, \
                   f"same multiset, {int((r!=d).sum())}/{n} positions differ{sh}"
    if n >= 4:
        bad = (r - d).abs() > (1e-3 + 1e-2 * r.abs())
        nb = int(bad.sum())
        if 0 < nb <= max(2, n // 4):
            dv = d[bad]
            if all(bool((((r - x).abs() <= 1e-3 + 1e-2 * r.abs()).any())) for x in dv):
                return "MISPLACED_ELEMS", "t7b", None, None, \
                       f"{nb}/{n} positions hold a value found elsewhere in ref"

    # --- t8 guarded SCALE
    m = r.abs() > 1e-3
    note = ""
    if bool(m.any()):
        ratio = d[m] / r[m]; ratio = ratio[torch.isfinite(ratio)]
        nz = int(ratio.numel())
        nondeg = float(r.std()) > 0.01 * max(1e-12, float(r.abs().mean())) if n > 1 else False
        if nz >= 2:
            rm = float(ratio.mean()); rsd = float(ratio.std()) if nz > 1 else 0.0
            ok = rsd < 0.05 * max(1e-9, abs(rm)) and abs(rm - 1) > 0.05
            if ok and nz >= 8 and nondeg:
                return f"SCALE:{rm:.4g}", "t8", None, None, f"n_ratio={nz} std/|mean|={rsd/max(1e-12,abs(rm)):.3g}"
            if ok:
                note = f"ratio={rm:.4g} consistent but VACUOUS (n_ratio={nz}{'' if nondeg else ', ref degenerate'})"

    # --- t9 numeric
    fin = rf & df
    adiff = float((d[fin] - r[fin]).abs().max()) if bool(fin.any()) else float("nan")
    rel_pt = float(((d[fin]-r[fin]).abs()/r[fin].abs().clamp_min(1e-12)).max()) if bool(fin.any()) else float("nan")
    rel_sc = adiff / max(rmax, 1e-12)
    mech = "NUMERIC" if ref.dtype.is_floating_point else "INT_WRONG_VALUE"
    det = f"max|delta|={adiff:.4g} |ref|max={rmax:.4g}"
    if note: det += "; " + note
    return mech, "t9", rel_pt, rel_sc, det


rows = []
for f in sorted(glob.glob(f"{DUMP}/*.pt")):
    D = torch.load(f, weights_only=False)
    cs = D["case"]
    m, tf, relp, rels, det = cls(D["ref"], D["dev"], D["node_vals"], D["target"])
    rows.append({"job": cs["job"], "out": cs["out"], "verdict": cs["verdict"],
                 "rec_mech": cs["mech"], "bucket": cs["bucket"], "tgt_op": cs["tgt_op"],
                 "trigger_ops": cs["trigger_ops"], "deleg_full": cs["deleg_full"],
                 "A_deleg": D["A_deleg"], "B_deleg": D["B_deleg"],
                 "A_n_ok": D["A_n_ok"], "B_n_div": D["B_n_div"],
                 "dtype": str(D["ref"].dtype).replace("torch.", ""),
                 "shape": str(tuple(D["ref"].shape)), "numel": D["ref"].numel(),
                 "mech": m, "test": tf,
                 "rel_pt": "" if relp is None else f"{relp:.4g}",
                 "rel_sc": "" if rels is None else f"{rels:.4g}",
                 "detail": det, "file": os.path.basename(f)})
json.dump(rows, open(f"{W}/reclass.json", "w"), indent=0)
cols = list(rows[0].keys())
with open(f"{W}/reclass.tsv", "w") as fh:
    fh.write("\t".join(cols) + "\n")
    for r in rows:
        fh.write("\t".join(str(r[c]).replace("\t", " ") for c in cols) + "\n")
from collections import Counter
def base(m): return m.split(":")[0]
print("n =", len(rows))
print("--- all:", Counter(base(r["mech"]) for r in rows).most_common())
for b in ("VALUE", "SCALE_ALIAS"):
    print(f"--- {b}:", Counter(base(r["mech"]) for r in rows if r["bucket"] == b).most_common())
print("--- VALUE by verdict:")
for v in ("COMPOSITIONAL", "SIBLING_DEPENDENT"):
    print("   ", v, Counter(base(r["mech"]) for r in rows if r["bucket"] == "VALUE" and r["verdict"] == v).most_common())
