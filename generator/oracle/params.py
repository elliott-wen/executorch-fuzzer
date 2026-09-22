"""params.py — how graphs in a corpus are shaped.

Its own module, and deliberately free of torch, so the supervising process can build a
worker's command line without importing the machinery that only the workers need. The
parent stays a supervisor; see mobile.generator.supervisor.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    nodes: int = 8                      # target real-op nodes per DAG
    leaf_prob: float = 0.3              # chance a port takes a fresh leaf over a producer
    out_alias_prob: float = 0.01        # chance an out= buffer aliases a live producer
    nonfinite_prob: float | None = None  # None → render.NONFINITE_PROB
    composable: bool = False            # include ops that only lower via decomposition
    retries: int = 8                    # build_graph attempts per graph before giving up
    target: str | None = None           # solve WITH a runtime's extra requirements; off by
                                        # default (generator/targets)

    def to_argv(self) -> list[str]:
        """These params as worker command-line flags — the inverse of add_arguments."""
        argv = ["--nodes", str(self.nodes), "--leaf-prob", str(self.leaf_prob),
                "--out-alias-prob", str(self.out_alias_prob), "--retries", str(self.retries)]
        if self.nonfinite_prob is not None:
            argv += ["--nonfinite-prob", str(self.nonfinite_prob)]
        if self.target:
            argv += ["--target", self.target]
        if self.composable:
            argv += ["--composable"]
        return argv

    @classmethod
    def add_arguments(cls, parser) -> None:
        """Register the flags to_argv emits, so the two stay in step."""
        parser.add_argument("--nodes", type=int, default=cls.nodes,
                            help=f"target real-op nodes per graph (default {cls.nodes})")
        parser.add_argument("--leaf-prob", type=float, default=cls.leaf_prob)
        parser.add_argument("--out-alias-prob", type=float, default=cls.out_alias_prob,
                            help="chance an out= buffer aliases a live producer, exercising "
                                 "the compiler's memory planning")
        parser.add_argument("--nonfinite-prob", type=float, default=None,
                            help="per-float-leaf chance of boundary values (default: render's)")
        parser.add_argument("--target", default=None,
                            help="also satisfy this runtime's extra requirements while "
                                 "solving (e.g. portable) — see generator/targets")
        parser.add_argument("--composable", action="store_true",
                            help="also emit ops that only lower by being decomposed")
        parser.add_argument("--retries", type=int, default=cls.retries,
                            help=f"build attempts per graph (default {cls.retries})")

    @classmethod
    def from_args(cls, args) -> "Params":
        return cls(nodes=args.nodes, leaf_prob=args.leaf_prob,
                   out_alias_prob=args.out_alias_prob, nonfinite_prob=args.nonfinite_prob,
                   composable=args.composable, retries=args.retries,
                   target=getattr(args, "target", None))
