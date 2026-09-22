"""eager — the target that requires nothing beyond PyTorch itself.

A target normally says how a runtime is NARROWER than eager. This one says it is not: no
axioms, no op delta, nothing added to any op's precondition. `--target eager` generates
exactly what `pytorch_constraints/` allows, which is every call PyTorch will accept.

Deliberately emptier than `portable`, which is already the most permissive real runtime:
portable still refuses complex and the float8 types, wants int64 indices, and pins a handful
of ops to narrower dtypes than eager needs. None of that applies here.

Two uses:

  * A BASELINE. Coverage lost to a target is only measurable against a run that lost none,
    so `--target eager` is what the others are compared with.
  * A NAMED way to say "no target". `--target` may be left off entirely for the same effect;
    naming it says the choice was made rather than forgotten, which matters in a script.

It stays empty on purpose. A rule that belongs to every runtime belongs in
`pytorch_constraints/` (if eager refuses the call) or in the blocklist (if the generator
cannot produce it soundly) — not here, where it would silently apply to a target whose whole
contract is that it applies nothing.
"""

from __future__ import annotations

# No `axioms` and no `allows`: both are optional, and their absence IS this module's content.
# targets.get() returns this module, targets.load_for() finds no axioms to merge, and
# targets.allows() finds no predicate and answers True for every op.
