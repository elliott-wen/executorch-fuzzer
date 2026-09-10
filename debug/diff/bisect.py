"""bisect.py — bug-localization on top of `ddmin`: which outputs / which operators own a divergence.

A lowered program returns SEVERAL outputs at once; one of them (the *target*) comes back wrong on
the backend. Two questions localize the bug, each answered by the same delta-debugging minimization
(`ddmin`) with a different injected predicate:

1. `bisect_output_set(target, others, target_diverges)` — **is the target wrong on its own, or only
   because of the OTHER outputs returned alongside it?** These are two *localities*, not root causes:
     * wrong on its own → **CONE_LOCAL**: the bug lives somewhere in the target's own operator chain
       (its dependency cone). Still an unknown mix — a single broken operator OR a graph-opt bug
       across operators along that chain. Do NOT read CONE_LOCAL as "operator bug".
     * only wrong in company → **SIBLING_DEPENDENT**: a whole-program optimization that misbehaves
       only when it must emit several outputs together (it re-plans memory / number-formats).
   The program keeps computing the same thing throughout — we only change *which results are handed
   back*, because that is what perturbs the compiler's memory plan and quantization choices.

2. `bisect_cone(target_node, ancestors, diverges_with_live)` — the SECOND minimization that finally
   splits a CONE_LOCAL failure into **OPERATOR** vs **COMPOSITIONAL** (graph-opt along the path), by
   minimizing which upstream operators must run LIVE vs. be replaced by correct constants.

A full localization is: `bisect_output_set` → if CONE_LOCAL, `bisect_cone`. Both hide the
lowering/broker/device behind the injected predicate, so this is reusable by any agent or harness.

Example:
    from mobile.gen.diff.bisect import bisect_output_set
    def target_diverges(returned):        # you implement: lower+run, compare target
        ...                               # return True iff the target still mismatches
    r = bisect_output_set(target="n7", others=["n3","n9","n10"], target_diverges=target_diverges)
    print(r.verdict, r.needed_siblings)   # CONE_LOCAL []  |  SIBLING_DEPENDENT ['n9']
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

from mobile.gen.diff.ddmin import ddmin

T = TypeVar("T")

# --- output-set verdicts (deliberately NOT "OPERATOR"; see module docstring) ------------------
CONE_LOCAL = "CONE_LOCAL"                # wrong with no sibling outputs → bug lives in target's cone
SIBLING_DEPENDENT = "SIBLING_DEPENDENT"  # only wrong when specific sibling outputs are co-returned
INCONCLUSIVE = "INCONCLUSIVE"            # shared: the divergence didn't hold up under minimization
                                         # (flaky / non-deterministic) → no evidence, do NOT guess a class


@dataclass
class OutputBisection:
    """Result of bisecting which co-returned outputs a target's divergence depends on.

    verdict          : CONE_LOCAL (needed_siblings == []) or SIBLING_DEPENDENT.
    needed_siblings  : minimal set of OTHER outputs that must be co-returned for the target to
                       stay wrong (empty ⇒ cone-local). This is a 1-minimal set from ddmin.
    """
    verdict: str
    needed_siblings: list


def bisect_output_set(
    target: T,
    others: Sequence[T],
    target_diverges: Callable[[Sequence[T]], bool],
) -> OutputBisection:
    """Find the minimal set of co-returned outputs the target's divergence needs.

    `target_diverges(returned_outputs) -> bool`: lower + run the program returning EXACTLY
    `returned_outputs`, and report whether `target` still diverges from the reference. The caller
    injects this — it hides the lowering / broker / device transport. `target` must appear in every
    call's set (this helper always includes it), so the predicate only ever varies the siblings.

    The full-set precondition is checked, not assumed: if `[target, *others]` no longer diverges on
    re-run, the failure is flaky and the result is INCONCLUSIVE (one extra predicate call).

    Returns OutputBisection:
      * verdict == CONE_LOCAL, needed_siblings == []  → the bug reproduces with the target alone.
        NB: cone-local is NOT "single operator" — the target's whole dependency chain still runs, so
        this is an unknown mix of single-op and along-the-path graph-opt bugs. Split it with
        `bisect_cone`.
      * verdict == SIBLING_DEPENDENT, needed_siblings == [...]  → a graph-opt bug triggered only
        when those sibling outputs are co-returned (memory-plan / quantization-plan interaction).
      * verdict == INCONCLUSIVE, needed_siblings == []  → the divergence didn't hold up (the full set
        didn't reproduce it, or ddmin pinned no required sibling): flaky / non-deterministic, no
        evidence for a class — do not guess one.
    """
    others = list(others)
    if not target_diverges([target, *others]):        # full set must reproduce, else it's flaky
        return OutputBisection(INCONCLUSIVE, [])
    if target_diverges([target]):
        return OutputBisection(CONE_LOCAL, [])
    needed = ddmin(others, lambda subset: target_diverges([target, *subset]))
    return OutputBisection(SIBLING_DEPENDENT, needed) if needed else OutputBisection(INCONCLUSIVE, [])


# ============================================================================================
# Second minimization: crack open a CONE_LOCAL failure into OPERATOR vs graph-opt-along-the-path.
# ============================================================================================
# Same ddmin, different (faithful) predicate. The elements are now the target's ANCESTOR operators
# in its dependency cone. "Keeping" an ancestor means computing it LIVE; "dropping" it means
# replacing its output with a CONSTANT baked from the *reference's* real intermediate value at that
# point. So the sub-graph the predicate builds is: target + the kept ancestors, with every input
# that crosses the cut replaced by a correct constant.
#
# Why baking from the REFERENCE (not the device) and why this is the faithful test:
#   * If the operator's own compiled kernel is wrong, it is wrong even when fed correct constant
#     inputs — it survives all the way down to the target computed alone. → OPERATOR.
#   * If instead the failure was a graph-opt / cross-op rescale, then computing the target alone
#     re-lowers it with its own fitting number-formats and the divergence DISAPPEARS; it only comes
#     back when specific upstream ops are computed live alongside it. → the failure is compositional.
#   This is exactly why naive single-op isolation was wrong as the ONLY test: it always cuts to the
#   target alone, so it silently reclassifies every compositional bug as "clean". bisect_cone does
#   not assume the target-alone case — it lets ddmin find the *minimal set of live ancestors* needed.

OPERATOR = "OPERATOR"              # diverges with target computed alone (inputs baked) → kernel wrong
COMPOSITIONAL = "COMPOSITIONAL"    # only diverges with ≥1 upstream op live → graph-opt along the path
# INCONCLUSIVE (shared, defined above): clean alone yet ddmin pinned no required live ancestor →
# no evidence either way (flaky / non-deterministic); do NOT guess a bug class.


@dataclass
class ConeBisection:
    """Result of cracking a cone-local failure into operator vs compositional graph-opt.

    verdict        : OPERATOR (needed_live == []) or COMPOSITIONAL or INCONCLUSIVE.
    needed_live    : minimal set of ancestor operators that must be computed LIVE (not baked to a
                     constant) for the target to stay wrong. Empty ⇒ the target op alone is wrong.
                     Non-empty ⇒ the divergence is born from computing those ops together (fusion /
                     cross-op number-format), i.e. a graph-optimization bug inside the path.
    """
    verdict: str
    needed_live: list


def bisect_cone(
    target_node: T,
    ancestors: Sequence[T],
    diverges_with_live: Callable[[Sequence[T]], bool],
) -> ConeBisection:
    """Split a CONE_LOCAL failure into OPERATOR vs COMPOSITIONAL graph-opt, reusing ddmin.

    `diverges_with_live(kept_ancestors) -> bool`: build a sub-graph that computes `target_node` plus
    the ancestors in `kept_ancestors` LIVE, with every input crossing the cut replaced by a CONSTANT
    baked from the reference's real value there; lower + run it; report whether `target_node` still
    diverges from the reference. The caller injects this (it owns the graph surgery + baking + device
    transport). `kept_ancestors == []` means "target computed directly on baked constant inputs"
    (faithful single-op isolation); the full `ancestors` set means the whole original cone.

    The full-cone precondition is checked, not assumed: if `diverges_with_live(list(ancestors))` is
    False on re-run, the failure is flaky and the result is INCONCLUSIVE (one extra predicate call).
    The baked constants MUST come from the reference, not the device (see the block comment above —
    that is what makes "operator" vs "compositional" meaningful).

    Returns ConeBisection:
      * verdict == OPERATOR, needed_live == []  → target diverges even computed alone on correct
        constant inputs → the operator's compiled kernel is genuinely wrong (survives isolation).
      * verdict == COMPOSITIONAL, needed_live == [...]  → target is fine alone but wrong once those
        upstream ops are computed live with it → a graph-optimization bug along the path (fusion /
        cross-op rescale), NOT a single-operator bug.
      * verdict == INCONCLUSIVE, needed_live == []  → the failure didn't hold up (full cone didn't
        reproduce it, or ddmin pinned no required live ancestor): flaky / non-deterministic, no
        evidence for either class — do not guess one.
    """
    ancestors = list(ancestors)
    if not diverges_with_live(ancestors):             # full cone must reproduce, else it's flaky
        return ConeBisection(INCONCLUSIVE, [])
    if diverges_with_live([]):
        return ConeBisection(OPERATOR, [])
    needed = ddmin(ancestors, diverges_with_live)
    return ConeBisection(COMPOSITIONAL, needed) if needed else ConeBisection(INCONCLUSIVE, [])


if __name__ == "__main__":
    # Self-test with pure predicates (no device) — verifies both bisections end to end.
    seen = bisect_output_set("t", ["a", "b", "c"],
                             lambda ret: "t" in ret and "b" in ret)  # target needs sibling 'b'
    assert seen.verdict == SIBLING_DEPENDENT and seen.needed_siblings == ["b"], seen
    lone = bisect_output_set("t", ["a", "b"], lambda ret: "t" in ret)  # wrong with target alone
    assert lone.verdict == CONE_LOCAL and lone.needed_siblings == [], lone
    opbug = bisect_cone("tgt", ["u1", "u2", "u3"], lambda kept: True)  # wrong even computed alone
    assert opbug.verdict == OPERATOR and opbug.needed_live == [], opbug
    comp = bisect_cone("tgt", ["u1", "u2", "u3"], lambda kept: "u2" in kept)  # needs u2 computed live
    assert comp.verdict == COMPOSITIONAL and comp.needed_live == ["u2"], comp
    # INCONCLUSIVE: the failure doesn't reproduce at all (full set/cone clean) → no guess
    assert bisect_output_set("t", ["a", "b"], lambda ret: False).verdict == INCONCLUSIVE
    assert bisect_cone("tgt", ["u1", "u2"], lambda kept: False).verdict == INCONCLUSIVE
    print("bisect self-test: OK")
