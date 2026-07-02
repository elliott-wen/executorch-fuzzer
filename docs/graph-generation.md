# 3. Graph generation

This is the heart of the fuzzer: how a random, **valid**, multi-op DAG is built.
Code: `gen/graph/build.py` (construction), `gen/ops/opnode.py` (per-op model),
`gen/graph/adapter.py` (port wiring), `gen/graph/ir.py` (the graph + emitter).

## The unit: an OpNode (Z3 model of one op)

Each op is an `OpNode` (`gen/ops/opnode.py`) wrapping its `approved_constraints/<symbol>/`
folder. It exposes:

- **`named_params`** — the op's full parameter list (name, type), in order.
- **`tensor_ports`** — the positional Tensor inputs (excluding `out=`) — the points
  where the op can consume another node's output.
- **`range_ports`** — ports carrying a *value* precondition (e.g. index/probability
  ports bounded via `<port>.data_lo/.data_hi`).
- **`generate(rng, pins=None)`** — ask Z3 to solve the op's constraints and return one
  concrete valid input set (`ConcreteArgs`), or `None` if UNSAT. With `pins`, some
  ports are forced to match a producer's spec (see *anchoring*).
- **`pin_for(port, tensor)`** — build the Z3 equality constraints that force `port` to
  equal a producer tensor's `(ndim, sizes, dtype)`.

The op's real preconditions live as Z3 axioms (`Not(And(*bad_clause))` over the
solver). So `generate` never yields inputs that would raise in eager — that is what
makes a downstream MISMATCH a genuine bug. The Z3 model itself is described in
[../gen/z3engine] (`gen/z3engine/model.py`): bounded ndim, Z3 Array theory for sizes/strides.

## The graph: GenGraph

`gen/graph/ir.GenGraph` is a functional DAG. Each node is `(op, slots)` where every
slot is one of:

- `('leaf', i)` — a fresh input tensor (index into `graph.leaves`),
- `('ref', nid)` — consume node `nid`'s output (a graph edge),
- `('const', v)` — a baked scalar / list / dtype / `out=` buffer.

`sink_ids()` returns the nodes nobody consumes — the graph's outputs. `emit()` renders
the whole thing as a standalone `.py` script defining `g(...)`, `LEAVES`, `_BACKEND`.

## Two-stage construction: seed, then grow

`build_graph(rng, opnodes, n_nodes, leaf_prob, seed_op, out_alias_prob)`:

```
seed one node            ← the only node GUARANTEED to be in the graph
then grow until n_nodes real op nodes:
    each grow step anchors a new node onto an existing producer
```

Only **real op nodes** count toward `n_nodes`. Adapter nodes (below) are free glue.

### Stage 1 — seed (`_seed`)

The seed is leaf-only (no upstream dependency), so it's the one node that always
lands. That makes the seed slot the lever for **op coverage**.

`seed_op`, when provided, *forces* which op occupies the seed slot. The `pregen`
driver passes it from a **deterministic round-robin schedule** so every op is seeded
an equal number of times instead of uniform-random (which structurally starves
constraint-heavy ops — see *coverage* below). `None` ⇒ legacy uniform pick.

### Stage 2 — grow (`_grow`)

Each grow step attaches one consumer node, up to **32 attempts**:

1. pick a candidate op (uniform — so op *combinations* stay diverse),
2. choose an **anchor port** (preferring a non-value-constrained one),
3. **anchor**: `op.generate(rng, op.pin_for(anchor_port, producer_output))` — Z3
   solves the new op's inputs *with one port pinned to an existing producer's spec*.
   SAT ⇒ the node provably wires onto the graph; UNSAT ⇒ try another triple.
4. validate via a meta probe (shape inference only — no real kernels run during build),
5. commit: wire the anchor as `('ref', producer)`, optionally connect other ports to
   producers via adapters, fill the rest with fresh leaves.

Everything during build runs on **meta tensors** (pure shape inference), so a crashing
op can't take down a generation worker — real crashes are deferred to the isolated
execution phase.

## Anchoring, in one picture

Producer emits `[4,8] float32`. To attach a `matmul`, pin one operand to `[4,8]`:

```
pins = [tv.ndim == 2, tv.sizes[0]==4, tv.sizes[1]==8, tv.dtype==float32]
op.generate(rng, pins)   # Z3 must now pick the OTHER operand as [8, k]
```

The pin propagates through the op's constraints; Z3 either finds consistent inputs for
the remaining ports (SAT — correct wiring by construction) or proves none exist
(UNSAT — reject this pairing). No trial-and-error on real tensors.

## The adapter ladder (port wiring)

When a grow node has *another* tensor port (beyond the anchor), the builder tries to
feed it from an existing producer rather than a fresh leaf — denser graphs exercise
more of the compiler. A producer rarely matches a port's `(shape, dtype)` exactly, so
`gen/adapter.py` inserts **glue**. `plan_connection` picks the most faithful option:

| plan | adapter op | when | effect |
|------|-----------|------|--------|
| `direct` | — | shape & dtype match | wire straight |
| `cast` | `_to_copy` | shape matches, dtype differs | change dtype |
| `broadcast` | `broadcast_to` | broadcastable (stretch size-1 axes) | stride-0 view, no new data |
| `reshape` | `view_copy` | same element count, different shape | rearrange (zero-copy alias in exir) |
| `slice` | `slice_copy` (chained per dim) | every dim `want ≤ src` | take a real sub-region (downsize) |
| `pad` | `constant_pad_nd` | every dim `want ≥ src` | zero-pad the tail (upsize) |
| `None` | — | nothing fits | use a fresh leaf instead |

Each carries a `_cast` variant (dtype also differs). Ordering is **most-transparent /
most-faithful first**: identity → dtype → stride-0 broadcast → numel-preserving reshape
→ real-data slice → zero-fabricating pad.

> **Why these and not others?** Every adapter here is verified **non-barrier** in the
> ExecuTorch backends — they're either zero-copy aliases in exir memory planning
> (`view_copy`) or delegated ops in the XNNPACK partitioner (`slice_copy`,
> `constant_pad_nd`). Glue that *fragments* a partition (e.g. `narrow`, `select`,
> `repeat`, `expand`) is deliberately avoided as glue — it would isolate the very ops
> we wired together. Those ops are still fuzzed as first-class nodes when chosen by the
> generator. See [backends.md](backends.md).

Value-constrained ports also get a `clamp` range-adapter so a producer's raw values
land in the port's solved `[lo, hi]`.

## `out=` aliasing

Ops with an `out=` buffer default to a fresh `torch.empty(...)` sized to the real
output. With probability `out_alias_prob` (default `0.1`), a grow node instead aliases
its `out=` onto an **existing producer of the exact same (shape, dtype)** — never one
the op also reads (input/output overlap is rejected by aten). This makes
`to_executorch`'s **memory-planning / buffer-reuse** path actually fire, instead of
every buffer being a fresh allocation. Tunable via `--out-alias-prob`.

## Coverage: the round-robin seed schedule

Uniform op selection is *uniform-then-feasibility-filtered*: easy ops sail through the
generate→export→lower gauntlet, hard ops fail somewhere and get silently re-rolled, so
they're under-represented or absent.

The fix lives in `net/pregen.py`: a fixed shuffle of the op list, shared by all workers
(so the global schedule is well-defined and reproducible), with each worker
phase-shifted by its own offset. For index `cur`:

```
seed_op = ops[sched_order[(sched_offset + cur) % len(ops)]]
```

Every op occupies the guaranteed seed slot an equal number of times; grow stays
uniform so combinations are still explored. It's a pure function of `cur`, so it's
resume/skip-safe and a `job_id` still reproduces its exact graph.

> Coverage means every op is **attempted** as the seed equally — not that every op
> reaches READY. An op that's generate-hard, export-broken, or backend-unsupported can
> still end at zero; the schedule turns "never tried" into "tried N times, measurably
> never READY," which is the signal you want.

## Knobs (generation)

| knob | where | meaning |
|------|-------|---------|
| `--nodes` | pregen / fleet | target **real** op nodes per graph (default 16) |
| `--leaf-prob` | pregen / fleet | chance a grow port uses a fresh leaf vs. connecting a producer (default 0.3) |
| `--out-alias-prob` | pregen / fleet | chance an `out=` buffer aliases a live producer (default 0.1) |
| `--seed` | pregen / fleet | global seed; with `--id`, fixes every graph |

See [cli-reference.md](cli-reference.md) for the full list.
