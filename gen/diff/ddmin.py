"""ddmin.py — Zeller-Hildebrandt delta-debugging minimization (the pure primitive).

Given a set of elements and a yes/no failure test, find the **smallest subset that still fails** —
like `git bisect`, but over a SET instead of a commit history. This module is *only* the algorithm:
no domain knowledge, no torch, no transport — you inject `still_fails(subset) -> bool`.

The bug-localization applications built on top of it live in `bisect.py`
(`bisect_output_set`, `bisect_cone`).

Example:
    from mobile.gen.diff.ddmin import ddmin
    ddmin([1, 2, 3, 4, 5, 6], lambda s: 2 in s and 5 in s)   # -> [2, 5]
"""
from __future__ import annotations

from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def ddmin(items: Sequence[T], still_fails: Callable[[list[T]], bool]) -> list[T]:
    """Return a 1-minimal sublist of `items` on which `still_fails` is True.

    `still_fails(subset) -> bool`: True iff the failure reproduces on `subset`. 1-minimal means
    removing any single remaining element makes it pass. Canonical Zeller-Hildebrandt ddmin:
    at each granularity `n`, first try shrinking to a single chunk (fast when the cause is small),
    then try dropping one chunk at a time (removes unneeded elements), then refine `n`.

    Precondition: `still_fails(list(items))` is True (the full set fails). If it isn't, the full
    set is returned unchanged — nothing to minimize against. Also assumes a meaningful failure
    needs ≥1 element (the empty set passes); ddmin narrows to a 1-minimal set, not below a
    singleton. Callers that care about the "fails with nothing at all" case handle it separately.
    Cost: O(n^2) predicate calls worst case; typically far fewer.
    """
    cur = list(items)
    if len(cur) <= 1 or not still_fails(cur):
        return cur
    n = 2
    while len(cur) > 1:
        chunks = [cur[i::n] for i in range(min(n, len(cur)))]
        # (a) can the failure be reproduced by a single chunk alone?  → narrow hard, reset n=2
        hit = next((ch for ch in chunks if ch and still_fails(ch)), None)
        if hit is not None:
            cur, n = hit, 2
            continue
        # (b) can we drop a chunk (keep its complement) and still fail?  → remove unneeded elements
        dropped = False
        for i in range(len(chunks)):
            comp = [x for j, ch in enumerate(chunks) if j != i for x in ch]
            if comp and still_fails(comp):
                cur, n = comp, max(n - 1, 2)
                dropped = True
                break
        if dropped:
            continue
        # (c) neither worked: refine granularity, or stop if already at element level
        if n >= len(cur):
            break
        n = min(len(cur), n * 2)
    return cur


if __name__ == "__main__":
    assert sorted(ddmin([1, 2, 3, 4, 5, 6], lambda s: 2 in s and 5 in s)) == [2, 5]  # pair culprit
    assert ddmin(list(range(20)), lambda s: 7 in s) == [7]                            # single culprit
    _r = ddmin([1, 2, 3, 4], lambda s: len(s) >= 3)                                   # any 3 is 1-minimal
    assert len(_r) == 3 and all(len([x for x in _r if x != d]) < 3 for d in _r)
    print("ddmin self-test: OK")
