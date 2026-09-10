"""Differential testing: lower a graph and diff eager vs. backend output.

Localizers, backend-agnostic and reusable:
  * `divergence` — dump every node's value and find the FIRST node that disagrees.
  * `ddmin` — the pure Zeller-Hildebrandt delta-debugging minimization primitive (no torch/broker).
  * `bisect` — bug localization built on ddmin: `bisect_output_set` (CONE_LOCAL vs SIBLING_DEPENDENT)
    and `bisect_cone` (OPERATOR vs COMPOSITIONAL). You inject the run/compare predicate.
  * `cone_predicate` — the graph surgery + faithful reference-baking that builds `bisect_cone`'s
    predicate from a corpus `.py` (transport injected).

Run as modules: `python -m mobile.gen.diff.divergence`, `... .ddmin`, `... .bisect`. Kept
import-light (no eager torch import at package load) so the CLI entry and lightweight consumers
don't pay for torch unless they touch the module.
"""
