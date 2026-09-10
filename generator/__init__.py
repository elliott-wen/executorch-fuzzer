"""generator — graph generation for the ExecuTorch differential fuzzer.

Bottom up:

- constraints: the Z3 model of ATen, and each op's precondition loaded from
               pytorch_constraints/. What a valid call looks like.
- concretize:  a satisfying Z3 model → concrete PyTorch call arguments. A faithful
               translator; it adds no policy of its own.
- graph:       the operator catalog and the graph builder. Picks ops, solves them,
               wires them into a DAG, and emits it as a standalone script.
- supervisor:  run work in disposable child processes and survive losing them. Generic:
               it knows nothing about graphs, only about jobs, acks and crashes.
- oracle:      graphs and their PyTorch reference, in bulk — `graph` under `supervisor`.

    from mobile.generator.graph import build_graph, load_ops

    ops = load_ops()
    graph = build_graph(rng, ops, n_nodes=8)
    source = graph.emit(seed=1234)     # defines g(*LEAVES); run it anywhere

Nothing here narrows generation to what a target handles well. An op is only ever tested
on the dtypes and shapes we generate for it, so a restriction applied at this layer
quietly decides that whole classes of input are correct by never asking about them.
Unsupported combinations are reported cleanly by the runtime instead.
"""
