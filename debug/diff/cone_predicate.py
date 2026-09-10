"""cone_predicate.py — build the `diverges_with_live` predicate that `bisect.bisect_cone` needs.

`bisect_cone` asks: of the operators upstream of a wrong target output, which must run LIVE (vs be
replaced by a correct constant) for the target to stay wrong? Answering it needs a function that,
given a set of ancestor ops to keep live, builds a sub-graph — target + kept ancestors, every input
crossing the cut baked to a CONSTANT from the reference's real value — and reports whether the
target still diverges on the backend.

This module builds that sub-graph source (the graph surgery + faithful baking) from an
auto-generated corpus `.py`. It parses with **Python's `ast`** and re-emits by AST transform +
`ast.unparse` — no regex, no string substitution — so it doesn't break when the emitter's spacing
or wrapping changes. It is transport-agnostic: you inject `run_and_diverges(src) -> bool` (build /
lower `src`, run on the device, compare the sole returned output to its eager reference, return True
iff they diverge). So the reusable surgery lives here; only the broker/FVP call is the caller's.

Usage:
    from mobile.gen.diff.cone_predicate import cone_localizer
    from mobile.gen.diff.bisect import bisect_cone
    ancestors, build_src, target = cone_localizer("corpus_v2/ethos-u/w0/w0_141.py", target_out=0)
    def diverges_with_live(kept):            # kept: subset of `ancestors`
        return run_and_diverges(build_src(kept))   # you implement run_and_diverges (lower+run+cmp)
    verdict = bisect_cone(target, ancestors, diverges_with_live)
    #   OPERATOR      → target wrong even computed alone on baked constants (kernel bug)
    #   COMPOSITIONAL → wrong only with verdict.needed_live ops live (graph-opt along the path)
"""
from __future__ import annotations

import ast
import copy

import torch

_FIRST_HELPER = (
    "def _first(r):\n"
    "    if isinstance(r, torch.Tensor): return r\n"
    "    if isinstance(r, (tuple, list)):\n"
    "        for x in r:\n"
    "            if isinstance(x, torch.Tensor): return x\n"
    "    return r\n"
)


def _load(path: str):
    """Parse a corpus graph `.py` structurally with `ast`.

    Returns (pre_stmts, leaves, node_val, order, deps, ret):
      pre_stmts : module statements BEFORE `def g` (imports, seed, _make/_first, leaf L* assigns)
      leaves    : the `def g(...)` argument names (the graph's input leaves)
      node_val  : {node_name -> the RHS expression AST}
      order     : node names in definition (topological) order
      deps      : {node_name -> set of node/leaf names it references} (structural, not textual)
      ret       : output node names, in return order
    """
    mod = ast.parse(open(path).read())
    pre: list[ast.stmt] = []
    gfunc: ast.FunctionDef | None = None
    for stmt in mod.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "g":
            gfunc = stmt
            break
        pre.append(stmt)
    if gfunc is None:
        raise ValueError(f"{path}: no `def g(...)` found")
    leaves = [a.arg for a in gfunc.args.args]
    node_val: dict[str, ast.expr] = {}
    order: list[str] = []
    ret: list[str] = []
    for stmt in gfunc.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)):
            name = stmt.targets[0].id
            node_val[name] = stmt.value
            order.append(name)
        elif isinstance(stmt, ast.Return):
            elts = stmt.value.elts if isinstance(stmt.value, ast.Tuple) else [stmt.value]
            ret = [e.id for e in elts if isinstance(e, ast.Name)]
    graph_names = set(order) | set(leaves)
    deps = {
        n: {x.id for x in ast.walk(node_val[n]) if isinstance(x, ast.Name) and x.id in graph_names} - {n}
        for n in order
    }
    return pre, leaves, node_val, order, deps, ret


def _host_values(pre, node_val, order) -> dict:
    """Run the full graph in eager on the host, capturing every node's reference value."""
    ns: dict = {}
    exec(compile(ast.Module(body=pre, type_ignores=[]), "<pre>", "exec"), ns)
    for n in order:
        assign = ast.fix_missing_locations(
            ast.Assign(targets=[ast.Name(id=n, ctx=ast.Store())], value=node_val[n]))
        exec(compile(ast.Module(body=[assign], type_ignores=[]), "<node>", "exec"), ns)
    return ns


def _ancestors(target: str, deps: dict) -> list[str]:
    """All n-node ancestors of `target` (leaves are excluded — they're always baked)."""
    seen: set[str] = set()
    stack = [d for d in deps.get(target, ()) if d in deps]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        stack += [d for d in deps.get(x, ()) if d in deps]
    return list(seen)


class _Rename(ast.NodeTransformer):
    """Replace cut-input Names with their baked replacement AST (a constant name / inlined literal)."""
    def __init__(self, rename: dict):
        self.rename = rename

    def visit_Name(self, node: ast.Name):
        repl = self.rename.get(node.id)
        return ast.copy_location(copy.deepcopy(repl), node) if repl is not None else node


def cone_localizer(graph_path: str, target_out: int):
    """Return `(ancestors, build_src, target)` for the given output index of a corpus graph.

    ancestors : list of the target op's live-or-bake-able upstream operators (n-nodes).
    build_src(kept) -> str : python source for a sub-graph that computes the target plus the
        ancestors in `kept` LIVE, with every input crossing the cut (a non-kept ancestor, or any
        leaf) replaced by a constant baked from the reference's real value. Returns just the target.
        `kept == []` ⇒ faithful single-op isolation of the target.
    target    : the target node name.
    """
    pre, _leaves, node_val, order, deps, ret = _load(graph_path)
    if target_out >= len(ret):
        raise IndexError(f"target_out {target_out} >= {len(ret)} outputs")
    target = ret[target_out]
    if target not in node_val:
        raise ValueError(f"target {target} is a leaf/output alias, not an operator node")
    ns = _host_values(pre, node_val, order)
    anc = _ancestors(target, deps)

    def build_src(kept) -> str:
        keptset = set(kept)
        emitted = [n for n in order if n == target or n in keptset]  # topo order, includes target
        emit_set = set(emitted)
        # cut inputs: names referenced by an emitted node that are not themselves emitted → bake them
        cut: list[str] = []
        for n in emitted:
            for r in deps[n]:
                if r not in emit_set and r not in cut:
                    cut.append(r)
        bake_lines: list[str] = []
        leaves: list[str] = []
        rename: dict[str, ast.expr] = {}
        for r in cut:
            v = ns[r]
            if isinstance(v, torch.Tensor):
                nm = f"B{len(leaves)}"
                bake_lines.append(
                    f"{nm} = torch.tensor({v.detach().cpu().flatten().tolist()}, "
                    f"dtype={v.dtype}).reshape({list(v.shape)})")
                leaves.append(nm)
                rename[r] = ast.Name(id=nm, ctx=ast.Load())
            else:
                rename[r] = ast.parse(repr(v), mode="eval").body  # inline non-tensor constant
        tr = _Rename(rename)
        body = [f"    {n} = {ast.unparse(tr.visit(copy.deepcopy(node_val[n])))}" for n in emitted]
        return (
            "import torch\n" + _FIRST_HELPER
            + ("\n".join(bake_lines) + "\n" if bake_lines else "")
            + f"LEAVES = [{', '.join(leaves)}]\n"
            + f"def g({', '.join(leaves)}):\n"
            + "\n".join(body) + "\n"
            + f"    return ({target},)\n"
        )

    return anc, build_src, target


def output_set_localizer(graph_path: str, target_out: int):
    """Return `(others, build_output_src, target)` for stage-1 (`bisect.bisect_output_set`).

    Unlike `cone_localizer`, this does NO baking — it keeps the REAL leaves and the full computation,
    and only changes WHICH outputs the graph returns. That is what perturbs the compiler's
    memory/quantization plan without touching the math, so it detects sibling-dependent graph-opt.

    others            : the other output names (the siblings of the target).
    build_output_src(returned) -> str : source for the graph returning the outputs in `returned`,
        emitted in the ORIGINAL return order (canonicalized) so a subset is a faithful sub-selection
        of the real program — reordering outputs would itself perturb the memory/quantization plan
        and hide the very sibling-dependence we test. `target` should be in `returned`.
    target            : the target node name.
    ret               : all output names in original return order (so callers can locate `target`'s
                        position in any canonicalized subset for comparison).
    """
    pre, leaves, node_val, order, deps, ret = _load(graph_path)
    if target_out >= len(ret):
        raise IndexError(f"target_out {target_out} >= {len(ret)} outputs")
    target = ret[target_out]
    others = [o for o in ret if o != target]
    pre_src = ast.unparse(ast.Module(body=pre, type_ignores=[]))

    def _cone(returned):
        seen: set[str] = set()
        stack = [r for r in returned if r in deps]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack += [d for d in deps[x] if d in deps]
        return seen

    def build_output_src(returned) -> str:
        sel = [o for o in ret if o in set(returned)]   # canonical: original return order
        keep = _cone(sel)
        body = "\n".join(f"    {n} = {ast.unparse(node_val[n])}" for n in order if n in keep)
        return (
            pre_src + "\n"
            + f"def g({', '.join(leaves)}):\n" + body + "\n"
            + f"    return ({', '.join(sel)},)\n"
            + f"LEAVES = [{', '.join(leaves)}]\n"
        )

    return others, build_output_src, target, ret
