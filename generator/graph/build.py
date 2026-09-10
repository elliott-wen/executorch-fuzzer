"""build.py — grow a random DAG out of the op catalog.

Each new node is attached in two steps:
  1. anchor — one tensor port is pinned directly to an existing producer's output, and the
     solver is asked to make the op's remaining arguments fit around it. This is what
     guarantees the node joins the graph instead of floating free.
  2. the rest — each remaining port either takes another producer through an adapter
     (probability 1 - leaf_prob) or becomes a fresh leaf.

A candidate is only committed once probe.meta_probe has confirmed the call is valid and
reported the shape it produces; nothing mutates the graph before that.
"""

from __future__ import annotations

import torch

from mobile.generator.graph.adapter import (
    make_broadcast, make_cast, make_clamp, make_pad, make_reshape, make_slice,
    plan_connection,
)
from mobile.generator.graph.dag import Graph
from mobile.generator.graph.probe import meta_probe

GROW_ATTEMPTS = 32          # candidate ops tried before giving up on extending the graph


# ── wiring ───────────────────────────────────────────────────────────────────────

def _clamp_into(source: int, graph: Graph, produced: dict, bounds) -> int:
    """Append a clamp so a producer's values land inside a port's required range. Shape
    and dtype are unchanged, so the recorded output spec carries over."""
    low, high = bounds
    node = graph.add(make_clamp(), [("ref", source), ("const", low), ("const", high)])
    produced[node] = produced[source]
    return node


def _connect(want, graph: Graph, produced: dict, producers: list, rng, bounds=None):
    """Try to wire some producer into a port that wants `want`'s shape and dtype,
    inserting adapters as needed. Returns a ('ref', nid) slot, or None when no producer
    connects — the caller then uses a fresh leaf."""
    want_shape, want_dtype = tuple(want.shape), want.dtype
    candidates = list(producers)
    rng.shuffle(candidates)

    for pid in candidates:
        meta = produced[pid]
        plan = plan_connection(meta.shape, meta.dtype, want_shape, want_dtype)
        if plan is None:
            continue
        try:
            source = pid
            if plan.endswith("_cast") or plan == "cast":
                meta = meta.to(want_dtype)
                source = graph.add(make_cast(), [("ref", source), ("const", want_dtype)])
                produced[source] = meta
            if plan.startswith("broadcast"):
                meta = torch.broadcast_to(meta, want_shape)
                source = graph.add(make_broadcast(),
                                   [("ref", source), ("const", list(want_shape))])
                produced[source] = meta
            elif plan.startswith("reshape"):
                meta = meta.reshape(want_shape)
                source = graph.add(make_reshape(),
                                   [("ref", source), ("const", list(want_shape))])
                produced[source] = meta
            elif plan.startswith("slice"):
                for dim in range(len(want_shape)):
                    if meta.shape[dim] == want_shape[dim]:
                        continue
                    meta = torch.ops.aten.slice_copy.Tensor(meta, dim, 0, want_shape[dim], 1)
                    source = graph.add(make_slice(), [
                        ("ref", source), ("const", dim), ("const", 0),
                        ("const", want_shape[dim]), ("const", 1)])
                    produced[source] = meta
            elif plan.startswith("pad"):
                pad = []
                for dim in reversed(range(len(want_shape))):
                    pad += [0, want_shape[dim] - meta.shape[dim]]
                meta = torch.ops.aten.constant_pad_nd.default(meta, pad, 0)
                source = graph.add(make_pad(), [("ref", source), ("const", pad), ("const", 0)])
                produced[source] = meta
            if bounds is not None:
                source = _clamp_into(source, graph, produced, bounds)
            return ("ref", source)
        except Exception:
            continue
    return None


def _slots_for_seed(op, args, out, ranges) -> list:
    """Argument slots for a seed node: every tensor port is a fresh leaf."""
    slots = []
    for (name, typ), value in zip(op.params, args):
        if name == "out" and isinstance(value, torch.Tensor):
            # Size the out= buffer to the real output, or a fullgraph compile rejects the
            # implied resize ("Shape mismatch with out= tensor variant").
            slots.append(("const", out))
        elif typ == "Tensor" and isinstance(value, torch.Tensor):
            # A value-constrained port bakes its solved, clamped values so the emitted
            # script doesn't re-roll out-of-range garbage through _make.
            slots.append(("baked_leaf" if name in ranges else "leaf", value))
        else:
            slots.append(("const", value))
    return slots


def _seed(rng, ops, graph: Graph, produced: dict, seed_op=None) -> int | None:
    """Plant the first node — leaves only. Returns its id, or None if it wouldn't build.

    `seed_op` forces which op takes this slot. The seed is the one node guaranteed to
    appear, so cycling it across the catalog is what gives every op equal representation:
    left to uniform choice, ops with tight preconditions fail feasibility more often and
    get starved (measured: 25 ops show up only when the slot is scheduled).

    One attempt only. Sampling is stochastic, so a retry can succeed where this failed —
    but how many attempts a graph is worth is the caller's policy, and duplicating it here
    would silently multiply against the caller's own retry loop.
    """
    op = seed_op if seed_op is not None else rng.choice(ops)
    args = op.generate(rng)
    if args is None:
        return None
    out = meta_probe(op, list(args.args))
    if out is None:
        return None
    ranges = getattr(args, "ranges", {}) or {}
    nid = graph.add(op, _slots_for_seed(op, args.args, out, ranges))
    produced[nid] = out
    return nid


def _attach(rng, ops, graph: Graph, produced: dict, leaf_prob: float,
            out_alias_prob: float) -> bool:
    """Attach one more node to the graph. True if one went in."""
    producers = list(produced.keys())

    for _ in range(GROW_ATTEMPTS):
        op = rng.choice(ops)
        if not op.tensor_ports:
            continue
        # Anchor on a port that doesn't constrain its data if possible, so a producer's
        # raw values don't land straight in an index/target port (handled with a clamp
        # below when every port is value-constrained).
        plain = [p for p in op.tensor_ports if p not in op.value_ports]
        anchor_port = rng.choice(plain or op.tensor_ports)
        anchor_pid = rng.choice(producers)

        args = op.generate(rng, op.pin_for(anchor_port, produced[anchor_pid]))
        if args is None:
            continue                       # infeasible against this anchor — try another
        out = meta_probe(op, list(args.args))
        if out is None:
            continue

        # Validated. Only past this point is the graph mutated.
        ranges = getattr(args, "ranges", {}) or {}
        producer_set = set(producers)
        read_directly: set[int] = set()    # producers this node reads via a storage-sharing ref
        out_slot = None
        slots = []

        for (name, typ), value in zip(op.params, args.args):
            if name == anchor_port:
                source = anchor_pid
                if name in ranges:
                    source = _clamp_into(source, graph, produced, ranges[name])
                else:
                    read_directly.add(source)
                slots.append(("ref", source))
            elif name == "out" and isinstance(value, torch.Tensor):
                out_slot = len(slots)
                slots.append(("const", out))
            elif typ == "Tensor" and isinstance(value, torch.Tensor):
                slot = None
                if rng.random() > leaf_prob:
                    slot = _connect(value, graph, produced, producers, rng,
                                    bounds=ranges.get(name))
                if slot is not None:
                    if slot[1] in producer_set:
                        read_directly.add(slot[1])   # adapters allocate, direct refs share
                    slots.append(slot)
                else:
                    slots.append(("baked_leaf" if name in ranges else "leaf", value))
            else:
                slots.append(("const", value))

        _maybe_alias_out(slots, out_slot, out, produced, producers, read_directly,
                         rng, out_alias_prob)
        nid = graph.add(op, slots)
        produced[nid] = out
        return True
    return False


def _maybe_alias_out(slots, out_slot, out, produced, producers, read_directly,
                     rng, probability) -> None:
    """Sometimes write into a live producer's buffer instead of a fresh one.

    This is what makes to_executorch's memory planning and buffer reuse actually fire —
    with every `out=` freshly allocated, that path is never exercised. The buffer must not
    be one this node also READS: overlapping input and output storage is rejected by aten.
    """
    if out_slot is None or rng.random() >= probability:
        return
    candidates = [p for p in producers
                  if p not in read_directly
                  and produced[p].shape == out.shape
                  and produced[p].dtype == out.dtype]
    if candidates:
        slots[out_slot] = ("ref", rng.choice(candidates))


def build_graph(rng, ops, n_nodes: int, leaf_prob: float = 0.3, seed_op=None,
                out_alias_prob: float = 0.01) -> Graph | None:
    """Grow a DAG of up to `n_nodes` real op nodes (adapters are glue and don't count).

    Returns the graph, or None if it couldn't even seed. With n_nodes=1 the growth loop is
    skipped and the result is a single-operator graph. A graph that gets stuck early is
    returned as-is rather than discarded — a shorter graph is still a valid test.
    """
    graph = Graph()
    produced: dict[int, torch.Tensor] = {}
    if _seed(rng, ops, graph, produced, seed_op=seed_op) is None:
        return None
    real_nodes = 1
    while real_nodes < n_nodes:
        if not _attach(rng, ops, graph, produced, leaf_prob, out_alias_prob):
            break
        real_nodes += 1
    return graph
