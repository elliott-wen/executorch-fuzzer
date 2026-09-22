"""generator — graph generation for the ExecuTorch differential fuzzer.

Bottom up:

- constraints: the Z3 model of ATen, and each op's precondition loaded from
               pytorch_constraints/. What a valid call looks like.
- concretize:  a satisfying Z3 model → concrete PyTorch call arguments. A faithful
               translator; it adds no policy of its own.
- targets:     what a RUNTIME demands on top of eager, per backend — extra constraints
               and its own op delta. Opt-in; eager accepts what these refuse.
- ops:         which operators we can emit, and how to get one valid call out of one.
               Composes the three above into Ops that generate themselves.
- graph:       given a set of ops, grow a DAG and emit it as a standalone script. Torch
               and nothing else — no constraints, no solver, no target.
- supervisor:  run work in disposable child processes and survive losing them. Generic:
               it knows nothing about graphs, only about jobs, acks and crashes.
- oracle:      graphs and their PyTorch reference, in bulk — `graph` under `supervisor`.

    from mobile.generator.ops import load_ops
    from mobile.generator.graph import build_graph

    ops = load_ops(target="portable")
    graph = build_graph(rng, ops, n_nodes=8)
    source = graph.emit(seed=1234)     # defines g(*LEAVES); run it anywhere

Narrowing generation to what a runtime handles well is OPT-IN, and off unless a target
is named. An op is only ever tested on the dtypes and shapes we generate for it, so every
restriction trades a class of input away for a higher run rate — and one applied by
default would quietly decide that class is correct by never asking about it. Naming a
target is that trade made deliberately, per backend, in one readable place; the untargeted
default keeps asking, and lets the runtime refuse cleanly at run time.
"""
