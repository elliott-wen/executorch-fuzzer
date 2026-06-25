"""
op_lookup.py — Map mangled C++ symbols to torch.ops.aten callables.

Symbol→op mapping is hard-coded in entry_points_data.py (no external TSV).


Usage:
    from mobile.engine.op_lookup import find_op_for_symbol

    op, op_name, dk = find_op_for_symbol(entry_sym)
    if op is not None:
        op(*args)
"""

from __future__ import annotations

from typing import Any

from mobile.engine.schema_overrides import schema_for_op

# Symbol → op mapping is HARD-CODED in entry_points_data.py (the mobile-allowlist
# subset, ~330 symbols) instead of read from entry_points.tsv — no external data
# file, self-contained and debuggable.
#   _BY_SYMBOL: {mangled → [(op_name, dispatch_key, demangled), ...]}
from mobile.engine.entry_points_data import BY_SYMBOL as _BY_SYMBOL


# ── Callable resolution ───────────────────────────────────────────────────────

def _resolve_callable(op_name: str) -> Any | None:
    """Return torch.ops.aten.<base>.<overload> (or <base>) for op_name."""
    try:
        import torch
        parts = op_name.split(".", 1)
        ns = getattr(torch.ops.aten, parts[0], None)
        if ns is None:
            return None
        if len(parts) == 2:
            return getattr(ns, parts[1], ns)
        # No overload suffix — prefer 'default' overload (has a schema) over the packet
        default_ol = getattr(ns, "default", None)
        return default_ol if default_ol is not None else ns
    except Exception:
        return None


def lookup_by_symbol(mangled: str) -> list[tuple[str, str]]:
    """Return all (op_name, dispatch_key) pairs for a mangled symbol."""
    return [(op_name, dk) for op_name, dk, _ in _BY_SYMBOL.get(mangled, [])]


def find_op_for_symbol(mangled: str) -> tuple[Any | None, str | None, str | None]:
    """
    Return (callable, op_name, dispatch_key) for the given mangled symbol.

    When multiple op_name/dispatch_key pairs share the same symbol, entries with
    a registered ATen schema are preferred — the schema signals a properly-named
    overload.  Falls back to TSV order if no entry has a schema.
    Returns (None, None, None) if the symbol is not found or no callable resolves.
    """
    entries = _BY_SYMBOL.get(mangled, [])
    if not entries:
        return None, None, None

    def _rank(entry: tuple[str, str, str]) -> int:
        return 0 if schema_for_op(entry[0]) is not None else 1

    for op_name, dk, demangled in sorted(entries, key=_rank):
        # Correct mismatched overload using the demangled C++ signature.
        # e.g. TSV maps special_xlog1py(Tensor,Scalar) to self_scalar but the
        # demangled name proves it's other_scalar (Tensor first).
        try:
            from z3gen.param_map import _fix_schema_overload, _schema_params, _demangled_arg_types
            schema = _schema_params(op_name)
            if schema:
                corrected = _fix_schema_overload(op_name, schema, demangled)
                if corrected is not schema:
                    # Schema changed — derive the corrected op_name from the overload
                    import torch as _torch
                    base = op_name.split(".")[0]
                    ns = getattr(_torch.ops.aten, base, None)
                    if ns:
                        for ov_name in getattr(ns, "_overload_names", []):
                            ov = getattr(ns, ov_name, None)
                            if ov is None: continue
                            try:
                                ov_params = [(a.name, str(a.type)) for a in ov._schema.arguments
                                             if not a.name.startswith("__")]
                                if ov_params == corrected:
                                    op_name = f"{base}.{ov_name}"
                                    break
                            except Exception:
                                pass
        except Exception:
            pass
        op = _resolve_callable(op_name)
        if op is not None:
            return op, op_name, dk
    return None, None, None
