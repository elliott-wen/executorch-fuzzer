"""xnnpack — what the XNNPACK delegate needs before it will TAKE a node.

This target is shaped by a different failure than portable's. A portable kernel that
refuses a call says so at run time and the graph is a visible SKIP. XNNPACK's partitioner
never refuses anything: a node it cannot take is simply left behind to run on the portable
kernels, the graph succeeds, and the run reports OK — having tested portable a second time
rather than XNNPACK at all.

That is not hypothetical. A 100,000-graph corpus generated against the portable rules and
lowered with --backend xnnpack delegated 85,581 of 2,649,689 ops (3.2%), and 57.6% of its
graphs delegated NOTHING. So the number this module moves is the delegation rate, not the
run rate, and the evidence for a rule is `lower.job.delegation`, not the skip log.

The rules apply to EVERY op, not only the ~50 the partitioner has a config for, because
delegation is decided AFTER to_edge decomposes: one native_layer_norm becomes mean/sub/mul/
rsqrt/add, every one of them delegatable. Measured on a corpus constrained only on the
configured ops, 129.7% of the candidate ATen nodes came back delegated — more than existed
— which is the decomposition products showing up. What they inherit is their producer's
DTYPE, so a graph seeded int64 stays int64 through every decomposition and delegates
nothing. Dtype is a whole-GRAPH property here, and a rule that skips most ops cannot set it.

Ops with a mandatory int64 or bool slot have no satisfiable call left under a float-only
rule. Those are named in `allows` below rather than left to collapse into an unsolvable
precondition: an op whose constraints have no model still LOADS, and then returns nothing
from generate() for the rest of the run — it stops appearing in graphs without anything
saying so, which reads as coverage. Excluding them by name reports the count instead.

Everything here is additive over portable, because a lowered .pte is a MIX: the ops the
partitioner took run XNNPACK kernels, the rest run portable ones and still need portable's
contracts.
"""

from __future__ import annotations

from typing import Any

from z3 import Implies, IntVal, Or, Select

from mobile.generator.constraints.model import FLOAT16, FLOAT32, MAX_DIM
from mobile.generator.targets import excluding, portable, tensors

#: XnnpackConfig._check_node_has_valid_dtype (xnnpack_config.py:228) — the whole set, for
#: a node that is not quantize/dequantize/qparam. Checked against EVERY tensor input and
#: EVERY tensor output, so one int64 index among an op's arguments loses the whole node.
DELEGATABLE_DTYPES = (FLOAT32, FLOAT16)


#: Ops that cannot satisfy the float-only rule above, so this target excludes them outright.
#: XNNPACK delegates none of them, so nothing is lost that was ever reachable. Found by
#: solving every op once with these axioms applied and keeping the UNSAT ones; re-derive the
#: same way after a change here. Grouped by what forces the non-float slot.
allows = excluding(
    # integral/bool operands by definition
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    "bitwise_left_shift", "bitwise_right_shift",
    # a bool or int64 RESULT: the output dtype check refuses the node
    "any", "argmax", "argmin", "topk", "max.dim_max", "min.dim_min",
    # int64 indices among the arguments
    "embedding", "gather", "index", "index_select", "scatter", "scatter_add",
    "masked_fill", "masked_scatter", "max_pool2d_with_indices_backward",
    # a bool condition
    "where",
    # complex, which portable pins and XNNPACK has never accepted
    "_conj_physical", "_fft_c2r", "view_as_real_copy",
)


def axioms(op_name: str, params: list[tuple[str, str]], variables: dict[str, Any]) -> list:
    """Portable's rules, plus what XNNPACK needs in order to delegate this node."""
    return portable.axioms(op_name, params, variables) + _delegatable(variables)


def _delegatable(variables: dict[str, Any]) -> list:
    """Steer a call into the envelope the partitioner accepts."""
    out, inputs = [], []
    for name, var in variables.items():
        for tensor in tensors(var):
            out.append(Or(*(tensor.dtype == IntVal(d) for d in DELEGATABLE_DTYPES)))
            if name != "out":
                inputs.append(tensor)

    for tensor in inputs:
        # "XNNPACK does not support empty tensors" (xnnpack_config.py:180). Written as the
        # linear "every active dimension is non-empty" rather than numel() >= 1, which the
        # model builds as a PRODUCT and hands the solver non-linear arithmetic.
        out += [Implies(IntVal(k) < tensor.ndim, Select(tensor.sizes, IntVal(k)) >= 1)
                for k in range(MAX_DIM)]

    # "does not support mixed input dtypes" (xnnpack_config.py:196): every tensor INPUT
    # must carry one dtype. `out` is exempt — the output is checked against the valid set
    # but never against the inputs, which is what keeps an fp32 -> fp16 node delegatable.
    out += [tensor.dtype == inputs[0].dtype for tensor in inputs[1:]]
    return out
