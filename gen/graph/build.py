"""graph_build.py — DAG construction (the feasibility-gated growth loop).

Producer output shapes are propagated with META tensors (pure shape inference,
no real kernels), so generation never runs real compute and a crashing op can't
take down a parallel generation worker. Real crashes surface later, in the
isolated RUN phase. We only ever read .dim()/.size()/.dtype off the meta result.

Growth wires each new node's ports, in priority order:
  1. anchor — pin ONE port directly to a producer (Z3 makes the op's other
     inputs fit); guarantees the node joins the existing graph.
  2. other ports — with prob (1 - leaf_prob), connect another producer via a
     natural fusion-transparent adapter (cast / broadcast); else a fresh leaf.
"""

from __future__ import annotations

import torch

from mobile.gen.graph.ir import GenGraph
from mobile.gen.graph.adapter import (
    make_cast, make_broadcast, make_reshape, make_slice, make_pad, make_clamp,
    plan_connection,
)

def _split_by_schema(op, args: list) -> tuple[list, dict]:
    """Split a flat arg list into (positional_args, kwarg_dict) using the op's JIT schema.

    ATen ops declare kwarg-only args (after `*` in the schema, e.g. `out=`) that
    must be passed as keyword arguments.  Passing them positionally raises TypeError.
    """
    try:
        schema = op._schema
        pos, kw = [], {}
        first_tensor_dtype = None
        for val, sa in zip(args, schema.arguments):
            if sa.kwarg_only:
                # Required kwarg-only args (no default, e.g. div.out_mode's
                # `rounding_mode`) must be passed even when None, or the call
                # raises "missing value for argument".
                required = not sa.has_default_value() if hasattr(sa, "has_default_value") else False
                if val is not None or required:
                    kw[sa.name] = val
            else:
                pos.append(val)
                if first_tensor_dtype is None and isinstance(val, torch.Tensor):
                    first_tensor_dtype = val.dtype
        # Synthesize missing kwarg-only Tensor outputs.  For meta-only structured
        # ops (e.g. norm.out) the `out` parameter is not in the arg_map — it is
        # reached via maybe_get_output in the meta — so it is never built and the
        # zip above skips it.  The `.out` overload requires it, so create an empty
        # out tensor (the op resizes it; dtype defaults to the first input's).
        n = len(args)
        for i, sa in enumerate(schema.arguments):
            if i < n:
                continue
            if sa.kwarg_only and "Tensor" in str(sa.type) and sa.name not in kw:
                kw[sa.name] = torch.empty(0, dtype=first_tensor_dtype or torch.float32)
        return pos, kw
    except Exception:
        return list(args), {}


def _to_meta(x):
    """Convert a real tensor to a meta tensor (shape/dtype only, no data). Recurses into
    list/tuple args so a `Tensor[]` port (e.g. cat/stack's `tensors`, or a `Tensor[] out=`)
    moves to meta ELEMENT-WISE — otherwise a list of CPU tensors stays on CPU while a
    synthesized meta out= triggers a cross-device meta<->cpu copy and the probe wrongly fails."""
    if isinstance(x, torch.Tensor):
        return x.to("meta")
    if isinstance(x, (list, tuple)):
        return type(x)(_to_meta(e) for e in x)
    return x


def _first_tensor(r):
    """First Tensor in an op's result (handles tuple/list outputs), or None."""
    if isinstance(r, torch.Tensor):
        return r
    if isinstance(r, (tuple, list)):
        for x in r:
            if isinstance(x, torch.Tensor):
                return x
    return None


def _functional_sibling(overload):
    """The functional overload sibling of a structured `.out` overload: same op packet,
    NO `out` argument, and a non-empty return. Structured `.out` variants like
    `split_copy.Tensor_out` return `()` and write into a `Tensor[] out=`, so probing them
    directly yields no tensor to read a shape from — the functional sibling
    (`split_copy.Tensor`) returns the outputs, which is all the probe needs. None if the
    packet has no such sibling."""
    try:
        packet = overload.overloadpacket
    except AttributeError:
        return None
    for nm in dir(packet):
        cand = getattr(packet, nm, None)
        sch = getattr(cand, "_schema", None)
        if sch is None or cand is overload:
            continue
        if sch.returns and not any(a.name == "out" for a in sch.arguments):
            return cand
    return None


def _meta_probe(op, args: list):
    """Run an op on meta versions of its args; return the meta output (first
    tensor) or None. Used to validate a node and get its output spec for pinning."""
    try:
        with torch.no_grad():
            margs = [_to_meta(a) for a in args]
            pos, kw = _split_by_schema(op.op, margs)
            # _split_by_schema synthesizes a real CPU `out=`; move it to meta so
            # the op runs all-meta instead of erroring on device mismatch.
            kw = {k: _to_meta(v) for k, v in kw.items()}
            out = _first_tensor(op.op(*pos, **kw))
            if out is not None:
                return out
            # Void / `Tensor[] out=` op (split/unbind/...): the `.out` overload returns
            # nothing, so there's no tensor to size the node from. Probe the functional
            # sibling (which returns the outputs) purely for shape inference — the graph
            # still emits and lowers the real `.out` overload downstream.
            sib = _functional_sibling(op.op)
            if sib is not None:
                pos2, kw2 = _split_by_schema(op.op, [_to_meta(a) for a in args])
                kw2.pop("out", None)
                return _first_tensor(sib(*pos2, **kw2))
            return None
    except Exception:
        return None


def _seed(rng, opnodes, graph, produced, seed_op=None):
    """Add a leaf-only seed node. Returns its node id, or None on failure.

    `seed_op`, when given, forces which op occupies the seed slot (the one node
    guaranteed to land in the graph) — the lever the coverage schedule pulls to
    exercise every op evenly instead of uniform-then-feasibility-filtered. When
    None, falls back to a uniform random pick (legacy behavior)."""
    op = seed_op if seed_op is not None else rng.choice(opnodes)
    c = op.generate(rng)
    if c is None:
        return None
    out = _meta_probe(op, list(c.args))
    if out is None:
        return None
    slots = []
    for (nm, ty), val in zip(op.named_params, c.args):
        if nm == "out" and isinstance(val, torch.Tensor):
            # Size the out= buffer to the real output (meta) so fullgraph compile
            # doesn't reject a resize ("Shape mismatch with out= tensor variant").
            slots.append(("const", out))
        elif ty == "Tensor" and isinstance(val, torch.Tensor):
            slots.append(("leaf", val))
        else:
            slots.append(("const", val))
    nid = graph.add(op, slots)
    produced[nid] = out
    return nid


def _clamp_into(src, graph, produced, range_bounds):
    """Append a clamp range-adapter so the producer's values land in [lo, hi]
    (the port's solved value range). Clamp preserves shape/dtype."""
    lo, hi = range_bounds
    cid = graph.add(make_clamp(), [("ref", src), ("const", lo), ("const", hi)])
    produced[cid] = produced[src]  # shape/dtype unchanged
    return cid


def _connect_port(want_tensor, graph, produced, producer_ids, rng, range_bounds=None):
    """Try to wire some producer into a port wanting want_tensor's (shape, dtype),
    inserting cast/broadcast adapter nodes as needed. If `range_bounds` is given
    (a value-constrained port), append a clamp range-adapter. Returns a ('ref', nid)
    slot, or None when no producer connects naturally (caller uses a leaf)."""
    want_shape = tuple(want_tensor.shape)
    want_dtype = want_tensor.dtype
    cands = list(producer_ids)
    rng.shuffle(cands)
    for pid in cands:
        pm = produced[pid]
        plan = plan_connection(pm.shape, pm.dtype, want_shape, want_dtype)
        if plan is None:
            continue
        try:
            src, src_meta = pid, pm
            if plan.endswith("_cast"):
                meta = src_meta.to(want_dtype)
                src = graph.add(make_cast(), [("ref", src), ("const", want_dtype)])
                produced[src] = meta
                src_meta = meta
            if plan in ("broadcast", "broadcast_cast"):
                meta = torch.broadcast_to(src_meta, want_shape)
                src = graph.add(make_broadcast(),
                                [("ref", src), ("const", list(want_shape))])
                produced[src] = meta
                src_meta = meta
            if plan in ("reshape", "reshape_cast"):
                # numel-preserving rearrange → view_copy (zero-copy alias in exir,
                # delegated static-reshape in XNNPACK). Cast already applied above.
                meta = src_meta.reshape(want_shape)
                src = graph.add(make_reshape(),
                                [("ref", src), ("const", list(want_shape))])
                produced[src] = meta
                src_meta = meta
            if plan in ("slice", "slice_cast"):
                # downsize: chain a stride-1 slice_copy per dim that must shrink
                # (delegated XNNStaticSlice; keeps a real sub-region of the data).
                for d in range(len(want_shape)):
                    if src_meta.shape[d] != want_shape[d]:
                        meta = torch.ops.aten.slice_copy.Tensor(
                            src_meta, d, 0, want_shape[d], 1)
                        src = graph.add(make_slice(), [
                            ("ref", src), ("const", d), ("const", 0),
                            ("const", want_shape[d]), ("const", 1)])
                        produced[src] = meta
                        src_meta = meta
            if plan in ("pad", "pad_cast"):
                # upsize: one constant_pad_nd zero-padding each dim's tail
                # (delegated XNNStaticConstantPad). pad list is last-dim-first.
                pad = []
                for d in reversed(range(len(want_shape))):
                    pad += [0, want_shape[d] - src_meta.shape[d]]
                meta = torch.ops.aten.constant_pad_nd.default(src_meta, pad, 0)
                src = graph.add(make_pad(),
                                [("ref", src), ("const", pad), ("const", 0)])
                produced[src] = meta
                src_meta = meta
            if range_bounds is not None:
                src = _clamp_into(src, graph, produced, range_bounds)
            return ("ref", src)
        except Exception:
            continue
    return None


def _grow(rng, opnodes, graph, produced, leaf_prob, out_alias_prob=0.01,
          attempts=32) -> bool:
    """Attach one consumer node (anchor + opportunistic adapter/leaf wiring)."""
    producer_ids = list(produced.keys())
    for _ in range(attempts):
        op = rng.choice(opnodes)
        if not op.tensor_ports:
            continue
        # Prefer a non-value-constrained port as the anchor, so a producer's raw
        # values don't land directly in an index/target port (handled below if
        # forced). Falls back to any port when all are value-constrained.
        plain = [p for p in op.tensor_ports if p not in op.range_ports]
        anchor_port = rng.choice(plain or op.tensor_ports)
        anchor_pid = rng.choice(producer_ids)
        c = op.generate(rng, op.pin_for(anchor_port, produced[anchor_pid]))
        if c is None:
            continue  # infeasible anchor for this sample → try another
        out = _meta_probe(op, list(c.args))
        if out is None:
            continue
        ranges = getattr(c, "ranges", {}) or {}
        # Validated — now commit wiring (graph mutated only past this point).
        producer_set = set(producer_ids)
        input_refs = set()      # producers THIS node reads via a direct (storage-sharing) ref
        out_slot_idx = None     # index of the out= slot, filled in opportunistically below
        slots = []
        for (nm, ty), val in zip(op.named_params, c.args):
            if nm == anchor_port:
                src = anchor_pid
                if nm in ranges:  # anchor is a value-constrained port → clamp it
                    src = _clamp_into(src, graph, produced, ranges[nm])
                else:
                    input_refs.add(src)  # direct anchor shares storage with the producer
                slots.append(("ref", src))
            elif nm == "out" and isinstance(val, torch.Tensor):
                # Default to a fresh buffer (sized to the real output so fullgraph
                # compile won't reject a resize — "Shape mismatch with out= tensor
                # variant"). May be aliased onto a live producer after the loop.
                out_slot_idx = len(slots)
                slots.append(("const", out))
            elif ty == "Tensor" and isinstance(val, torch.Tensor):
                slot = None
                if rng.random() > leaf_prob:
                    slot = _connect_port(val, graph, produced, producer_ids, rng,
                                         range_bounds=ranges.get(nm))
                if slot is not None and slot[1] in producer_set:
                    input_refs.add(slot[1])  # direct producer reuse (adapters → fresh storage)
                slots.append(slot if slot is not None else ("leaf", val))
            else:
                slots.append(("const", val))
        # out= aliasing: with prob out_alias_prob, write into an existing producer of
        # the EXACT same (shape, dtype) instead of a fresh buffer — this is what makes
        # to_executorch's memory-planning / buffer-reuse path actually fire. Never alias
        # onto a tensor this op also reads (input/output memory overlap → aten rejects).
        if out_slot_idx is not None and rng.random() < out_alias_prob:
            cands = [p for p in producer_ids
                     if p not in input_refs
                     and produced[p].shape == out.shape
                     and produced[p].dtype == out.dtype]
            if cands:
                slots[out_slot_idx] = ("ref", rng.choice(cands))
        nid = graph.add(op, slots)
        produced[nid] = out
        return True
    return False


def build_graph(rng, opnodes, n_nodes: int, leaf_prob: float = 0.3,
                seed_op=None, out_alias_prob: float = 0.01) -> GenGraph | None:
    """Grow a functional DAG of up to n_nodes (real op nodes; adapters are extra).
    Returns the graph (≥1 op node), or None if it couldn't even seed. With
    n_nodes=1 the grow loop is skipped and you get a single-operator graph.

    `seed_op` pins the seed slot (see `_seed`); grow nodes stay uniform-random so
    op *combinations* are still explored. None ⇒ fully-uniform legacy behavior.
    `out_alias_prob` is the chance a grow node's out= buffer aliases a live producer
    (exercises compiler memory-planning) instead of allocating fresh (see `_grow`)."""
    graph = GenGraph()
    produced: dict[int, torch.Tensor] = {}
    if _seed(rng, opnodes, graph, produced, seed_op=seed_op) is None:
        return None
    # Count only real op nodes toward the budget (adapters are glue, not the point).
    real_nodes = 1
    while real_nodes < n_nodes:
        if not _grow(rng, opnodes, graph, produced, leaf_prob, out_alias_prob):
            break  # stuck — return whatever we managed to build
        real_nodes += 1
    return graph
