#!/usr/bin/env python3
"""deep_dive.py — detailed root-cause dump for ONE corpus graph on the QNN emulator.

For the first divergent node, prints the exact op call (with args from the .py source),
the input node values that triggered it, and the eager-vs-QNN output tensors (dtype, shape,
nan/inf counts, sample values, max delta). Grounds the "why" behind each flagged bug.

Usage: deep_dive.py <corpus.py>   (caller sets the QNN env, see localize_one.py)
"""
import re
import sys

import torch

import mobile.gen.diff.divergence as D

_orig = D._resolve_lower
D._resolve_lower = lambda b: (_orig(b)[0], True) if b in ("qualcomm", "qnn") else _orig(b)


def _summ(t):
    if not isinstance(t, torch.Tensor):
        return f"{type(t).__name__}={t!r}"
    flat = t.flatten().to(torch.float64) if t.is_floating_point() or t.dtype in (
        torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool) else None
    nan = inf = "-"
    if t.is_floating_point():
        nan = int(torch.isnan(t).sum()); inf = int(torch.isinf(t).sum())
    sample = t.flatten()[:6].tolist()
    return f"dtype={t.dtype} shape={tuple(t.shape)} nan={nan} inf={inf} sample={sample}"


def _srcline(path, node):
    rx = re.compile(rf"^\s*{re.escape(node)}\s*=")
    for ln in open(path):
        if rx.match(ln):
            return ln.strip()
    return "(source line not found)"


def main() -> int:
    path = sys.argv[1]
    _, names, ops_map, _, _ = D.parse_corpus_graph(path)
    mod = D._load_allnode_module(path, names)
    leaves = list(mod.LEAVES)
    eager = list(mod.g(*leaves))
    node_ops = [ops_map[n] for n in names]
    tensor_idx = [i for i, v in enumerate(eager) if isinstance(v, torch.Tensor)]

    class W(torch.nn.Module):
        def forward(self, *a):
            outs = mod.g(*a)
            return tuple(outs[i] for i in tensor_idx)

    lower, _ = D._resolve_lower("qualcomm")
    backend_t = D.run_all_nodes_on_host(W(), leaves, lower)
    backend = list(eager)
    for j, i in enumerate(tensor_idx):
        backend[i] = backend_t[j]

    div = D.first_divergence(eager, backend, names, node_ops)
    print(f"# {path}")
    print(f"LEAVES: " + " | ".join(_summ(l) for l in leaves))
    if div is None:
        print("no divergence")
        return 0
    i = div.index
    node = names[i]
    print(f"\nFIRST DIVERGENCE: node={node} op={div.op} kind={div.kind} index={i}")
    print(f"  src: {_srcline(path, node)}")
    e, b = eager[i], backend[i]
    print(f"  eager : {_summ(e)}")
    print(f"  qnn   : {_summ(b)}")
    if isinstance(e, torch.Tensor) and isinstance(b, torch.Tensor) and e.shape == b.shape:
        if e.is_floating_point():
            ef, bf = e.flatten().double(), b.flatten().double()
            finite = torch.isfinite(ef) & torch.isfinite(bf)
            if finite.any():
                d = (ef[finite] - bf[finite]).abs().max().item()
                print(f"  max|delta| (finite positions) = {d:.6g}")
            print(f"  eager nan/inf positions not matched by qnn: "
                  f"{int(((~torch.isfinite(ef)) & torch.isfinite(bf)).sum())}")
    # show the inputs feeding the divergent op (producer nodes referenced in its src line)
    src = _srcline(path, node)
    refs = [n for n in names if re.search(rf"\b{n}\b", src.split('=', 1)[1]) and n != node]
    for r in refs[:4]:
        ri = names.index(r)
        print(f"  input {r}: {_summ(eager[ri])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
