"""Op catalog: which operators exist and which are allowed for generation.

- opnode:    the OpNode model + load_opnodes (discover the symbolic op set)
- blocklist: is_blocked (ops never generated)
- allowlist: is_executorch_op (ops the ExecuTorch portable runtime supports)
- runnable:  load_runnable_opnodes (the generator-ready subset)
"""
from __future__ import annotations

from .opnode import OpNode, load_opnodes
from .runnable import load_runnable_opnodes
from .blocklist import is_blocked
from .allowlist import is_executorch_op

__all__ = [
    "OpNode",
    "load_opnodes",
    "load_runnable_opnodes",
    "is_blocked",
    "is_executorch_op",
]
