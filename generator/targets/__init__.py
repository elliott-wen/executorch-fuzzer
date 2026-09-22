"""targets — what a specific RUNTIME requires, on top of what PyTorch itself requires.

`pytorch_constraints/` is extracted from PyTorch and says when an op RAISES IN EAGER. A
backend is usually stricter: ExecuTorch's portable kernels take int64 indices where eager
takes int32 or int64, require `native_group_norm`'s N and C to equal the input's first two
dimensions where eager only checks the product, and refuse `non_blocking=True` outright. A
graph that satisfies eager can therefore lower cleanly and still be refused at run time.

A target module says what that runtime additionally requires, as Z3 the solver satisfies
ALONGSIDE the op's real precondition. That is the point of doing it here rather than
rewriting values afterwards: a post-solve substitution can contradict a relation the solver
has just satisfied (canCast, promote_types, same-as between arguments), and the op then
rejects its own inputs. Constraints cannot contradict anything, because the solver holds all
of them at once.

    from mobile.generator.targets import load_for

    constraints = load_for(symbol, op_name, "portable")   # target name, or None for off

One module per backend, declaring either or both of the two ways a runtime is narrower
than eager:

  axioms(op_name, params, variables)   calls it refuses — solved WITH the op's precondition
  allows(op_name)                      operators it cannot run at all — the op delta

Prefer `axioms` wherever it can express the limit. A rule keeps the op and steers the
solver away from the calls that fail; an exclusion drops the op and every bug it would have
found with it. `allows` is for ops the runtime genuinely has no kernel or delegate for,
where there is no valid call left to generate.

Two kinds of extra constraint live in different places, because they behave differently:

  constraints/  an ATen precondition, including any the extractor missed. Always on — eager
                itself raises, so the call is simply invalid. That package knows nothing
                about runtimes, which is why this one is not inside it.
  targets/      a restriction one runtime adds on top. Opt-in — eager accepts these calls
                happily; only the target refuses them.

A target is chosen independently of the backend a graph is later lowered to, and that is
deliberate: generating against `portable` and lowering through `xnnpack` is the normal
case, since whatever the partitioner declines falls back to portable kernels anyway.

Generated files are never hand-edited: a module here survives re-extracting
pytorch_constraints/, which a HAND-ADDED marker inside a generated file does not.
"""

from __future__ import annotations

import dataclasses
import importlib
from types import ModuleType
from typing import Any, TYPE_CHECKING

from mobile.generator.constraints.model import OptVar, TensorListVar, TensorVar

if TYPE_CHECKING:
    from mobile.generator.constraints.loader import OpConstraints

#: backend name -> module in this package. Add a file, add a line.
_TARGETS: dict[str, str] = {
    "portable": "portable",
    "xnnpack": "xnnpack",
    "vulkan": "vulkan",
    "vgf": "vgf",
    "qualcomm": "qualcomm",
    "samsung": "samsung",
    "cadence": "cadence",
    "coreml": "coreml",
    "openvino": "openvino",
    "mediatek": "mediatek",
    "nxp": "nxp",
    "cuda": "cuda",
    "cortex-m": "cortexm",
    "ethos-u": "ethosu",
    "webgpu": "webgpu",
    "mlx": "mlx",
    "eager": "eager",           # adds nothing — the baseline the others are measured against
}


def get(name: str | None) -> ModuleType | None:
    """The target module for `name`, or None for no extra constraints — the default.

    The module IS the target. It may expose either or both of:

        axioms(op_name, params, variables) -> list   what this runtime demands of a CALL
        allows(op_name) -> bool                      which operators it will run at all

    Both are optional and both default to "adds nothing / refuses nothing", so a new target
    can start as an empty module and grow. Nothing is cached here because importlib already
    memoises the module, and a dict lookup is the rest of the work.
    """
    if not name:
        return None
    module_name = _TARGETS.get(name)
    if module_name is None:
        raise KeyError(f"no target constraints for {name!r}; "
                       f"have: {', '.join(sorted(_TARGETS))}")
    return importlib.import_module(f"{__name__}.{module_name}")


def tensors(var: Any):
    """Every TensorVar reachable from a model variable, through optionals and lists.

    Shared because a rule about tensors is the commonest kind a target writes, and each
    one would otherwise re-walk OptVar/TensorListVar itself.
    """
    if isinstance(var, TensorVar):
        yield var
    elif isinstance(var, OptVar):
        yield from tensors(var.value)
    elif isinstance(var, TensorListVar):
        for item in var.tensors:
            yield from tensors(item)


def allows(op_name: str, target: str | None = None) -> bool:
    """Whether `target` will run `op_name` at all — a target's OP DELTA.

    Asked before an op's constraints are loaded, which is the only reason it is worth
    having: excluding an op the runtime cannot run saves the expensive load as well as the
    graphs that would have been wasted on it.

    This narrows generation, so it hides whatever bugs those ops would have found — an op
    is only ever tested on what we generate for it. Excluding one is a claim that the
    runtime CANNOT run it, not that it currently fails; a kernel that refuses some calls
    belongs in `axioms`, where the solver avoids exactly those calls and keeps the rest.
    """
    module = get(target)
    if module is None:
        return True
    predicate = getattr(module, "allows", None)
    return predicate is None or predicate(op_name)


def excluding(*names: str):
    """An `allows` that refuses `names` — each matching an op_name exactly, its base name
    (before the first '.', so every overload of a family), or as a prefix when it ends '*'.

    A target with a plain list of unsupported ops assigns this and writes no logic:

        allows = excluding("_fft_r2c", "convolution*")
    """
    exact = frozenset(n for n in names if not n.endswith("*"))
    prefixes = tuple(n[:-1] for n in names if n.endswith("*"))

    def allows(op_name: str) -> bool:
        return not (op_name in exact
                    or op_name.split(".", 1)[0] in exact
                    or (prefixes and op_name.startswith(prefixes)))

    return allows


def known() -> list[str]:
    return sorted(_TARGETS)


def load_for(symbol: str, op_name: str | None = None,
             target: str | None = None) -> "OpConstraints | None":
    """`symbol`'s constraints, with `target`'s extra requirements merged into `axioms`.

    Composed here rather than inside `loader.load`, which is keyed on the symbol alone: a
    target is keyed on the OP NAME and chosen per run, so folding it in would broaden that
    function's contract to something it cannot answer from its key. Keeping the two apart is
    also what lets `constraints/` stay a description of eager and nothing else.

    The result is an ordinary OpConstraints, so nothing downstream needs to know a target
    was involved — no extra field on Op, no extra argument to build_solver.
    """
    from mobile.generator.constraints.loader import load

    constraints = load(symbol)
    module = get(target)
    if constraints is None or module is None or op_name is None:
        return constraints
    build = getattr(module, "axioms", None)          # optional, like `allows`
    extra = build(op_name, constraints.params, constraints.vars) if build else None
    if not extra:
        return constraints
    # Shares `vars`: the target's axioms are written against those same variables, so the
    # replacement carries the originals rather than rebuilding them.
    return dataclasses.replace(constraints, axioms=[*constraints.axioms, extra])
