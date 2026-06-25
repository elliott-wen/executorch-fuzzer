"""
verify.py — Verification for generated get_constraints() functions (agent pipeline).

Level 1 (Z3 satisfiability check, always runs):
    - Build a solver with basic domain axioms; hard timeout = Z3_TIMEOUT_MS.
    - Assert And(*constraints).
    - sat     → extract a concrete witness (the "bad input").
    - unsat   → constraints are contradictory; retry with error.
    - unknown → solver timed out; retry with a simplification hint.

Level 2 (PyTorch runtime check, only when no coarse-grained free Bools):
    - Build concrete Python/PyTorch arguments from the Z3 witness.
    - Call the ATen op via torch.ops.aten.<name>.
    - RuntimeError → TRIGGERED (good).
    - No error      → NOT TRIGGERED (constraints too weak or wrong overload).
    - unknown (Z3 timeout during witness search) → inconclusive; accepted.

Usage as a standalone script:
    python3 agent/verify.py <constraint.py> <ll_file>
"""

from __future__ import annotations

import argparse
import os
import select
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pickle

sys.path.insert(0, str(Path(__file__).parent))

from z3 import And, Or, Not, If, Implies, Select, IntVal, BoolVal, Solver, sat, unknown, is_const

from mobile.gen.concretize import build_call_args
from mobile.engine.model import (
    _ARG_BUILDERS, OptVar,
    MAX_DIM,
    DEVICE_CPU, DEVICE_CUDA, DEVICE_XPU, DEVICE_MPS, DEVICE_META,
    DEVICE_PRIVATEUSE1,
    TensorVar, ScalarVar, IntArrayVar, TensorListVar, GeneratorStub,
    UINT8, INT8, INT16, INT32, INT64,
    FLOAT16, FLOAT32, FLOAT64,
    COMPLEX32, COMPLEX64, COMPLEX128,
    BOOL, BFLOAT16,
    BITS1X8, BITS2X4, BITS4X2, BITS8, BITS16,
    FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT8_E5M2FNUZ, FLOAT8_E4M3FNUZ, FLOAT8_E8M0FNU,
    UINT16, UINT32, UINT64,
    LAYOUT_STRIDED,
    MEMORY_FORMAT_CONTIGUOUS, MEMORY_FORMAT_PRESERVE,
    MEMORY_FORMAT_CHANNELS_LAST, MEMORY_FORMAT_CHANNELS_LAST_3D,
    check_memory_format, check_resizable, check_scaling,
    check_overlap, check_internal_overlap, check_name, check_contiguous,
    check_deterministic, check_deterministic_fill, check_sdp_backend,
    promote_types,
    result_type,
    common_dtype,
    is_same,
    same_shape,
    is_expandable_to,
    broadcastable,
    NO_NAME,
    are_names_equal,
    maybe_wrap_dim,
    canCast,
    isBitsType,
    detail_scalar_type,
    is_floating_type,
    is_integral_type,
    is_complex_type,
    is_reduced_floating_type,
    toRealValueType,
    toComplexType,
    z3_div_floor,
)
from mobile.engine.op_lookup import find_op_for_symbol


# ── Z3 solver timeout ────────────────────────────────────────────────────────

# Z3 wall-clock budget per solver.check() call (milliseconds).
Z3_TIMEOUT_MS = 10_000


def _make_solver() -> Solver:
    s = Solver()
    s.set("timeout", Z3_TIMEOUT_MS)
    return s


# ── exec() sandbox globals ────────────────────────────────────────────────────

_EXEC_GLOBALS: dict[str, Any] = {
    # model types and bounds
    "MAX_DIM": MAX_DIM,
    # DeviceType constants
    "DEVICE_CPU": DEVICE_CPU, "DEVICE_CUDA": DEVICE_CUDA,
    "DEVICE_XPU": DEVICE_XPU, "DEVICE_MPS": DEVICE_MPS,
    "DEVICE_META": DEVICE_META, "DEVICE_PRIVATEUSE1": DEVICE_PRIVATEUSE1,
    "TensorVar": TensorVar, "ScalarVar": ScalarVar,
    "IntArrayVar": IntArrayVar, "OptVar": OptVar,
    "TensorListVar": TensorListVar, "GeneratorStub": GeneratorStub,
    # ScalarType constants
    "UINT8": UINT8, "INT8": INT8, "INT16": INT16,
    "INT32": INT32, "INT64": INT64,
    "FLOAT16": FLOAT16, "FLOAT32": FLOAT32, "FLOAT64": FLOAT64,
    "COMPLEX32": COMPLEX32, "COMPLEX64": COMPLEX64, "COMPLEX128": COMPLEX128,
    "BOOL": BOOL, "BFLOAT16": BFLOAT16,
    # Bits types
    "BITS1X8": BITS1X8, "BITS2X4": BITS2X4, "BITS4X2": BITS4X2,
    "BITS8": BITS8, "BITS16": BITS16,
    # Float8 types
    "FLOAT8_E5M2": FLOAT8_E5M2, "FLOAT8_E4M3FN": FLOAT8_E4M3FN,
    "FLOAT8_E5M2FNUZ": FLOAT8_E5M2FNUZ, "FLOAT8_E4M3FNUZ": FLOAT8_E4M3FNUZ,
    "FLOAT8_E8M0FNU": FLOAT8_E8M0FNU,
    # Extended unsigned integers
    "UINT16": UINT16, "UINT32": UINT32, "UINT64": UINT64,
    # Layout constants
    "LAYOUT_STRIDED": LAYOUT_STRIDED,
    # MemoryFormat constants
    "MEMORY_FORMAT_CONTIGUOUS": MEMORY_FORMAT_CONTIGUOUS,
    "MEMORY_FORMAT_PRESERVE": MEMORY_FORMAT_PRESERVE,
    "MEMORY_FORMAT_CHANNELS_LAST": MEMORY_FORMAT_CHANNELS_LAST,
    "MEMORY_FORMAT_CHANNELS_LAST_3D": MEMORY_FORMAT_CHANNELS_LAST_3D,
    # Coarse-grained free Bools
    "check_memory_format": check_memory_format,
    "check_resizable": check_resizable,
    "check_scaling": check_scaling,
    "check_overlap": check_overlap,
    "check_internal_overlap": check_internal_overlap,
    "check_name": check_name,
    "check_contiguous": check_contiguous,
    "check_deterministic": check_deterministic,
    "check_deterministic_fill": check_deterministic_fill,
    "check_sdp_backend": check_sdp_backend,
    # Type promotion and TensorIterator helpers
    "promote_types": promote_types,
    "result_type": result_type,
    "common_dtype": common_dtype,
    # Tensor identity, shape and name comparison
    "is_same": is_same,
    "same_shape": same_shape,
    "is_expandable_to": is_expandable_to,
    "broadcastable": broadcastable,
    "NO_NAME": NO_NAME,
    "are_names_equal": are_names_equal,
    # Dimension wrapping
    "maybe_wrap_dim": maybe_wrap_dim,
    # Cast and dtype helpers
    "canCast": canCast,
    "isBitsType": isBitsType,
    "detail_scalar_type": detail_scalar_type,
    "is_floating_type": is_floating_type,
    "is_integral_type": is_integral_type,
    "is_complex_type": is_complex_type,
    "is_reduced_floating_type": is_reduced_floating_type,
    "toRealValueType": toRealValueType,
    "toComplexType": toComplexType,
    # Z3 helpers
    "And": And, "Or": Or, "Not": Not, "If": If, "Implies": Implies,
    "Select": Select, "IntVal": IntVal, "BoolVal": BoolVal,
    "z3_div_floor": z3_div_floor,
}


# ── build_vars ────────────────────────────────────────────────────────────────

def build_vars(named_params: list[tuple[str, str]]) -> dict[str, Any]:
    """
    Create Z3 symbolic variables for each (name, type) pair in named_params.

    Adapted from llm/extract._build_vars.
    """
    vars_: dict[str, Any] = {}
    for name, typ in named_params:
        builder = _ARG_BUILDERS.get(typ)
        vars_[name] = builder(name) if builder else None
    return {k: v for k, v in vars_.items() if v is not None}


# ── exec_constraints ─────────────────────────────────────────────────────────

def exec_constraints(code: str, named_vars: dict[str, Any]) -> list:
    """
    Execute generated constraint code and return get_constraints(**named_vars).

    Runs in an isolated namespace seeded with _EXEC_GLOBALS and the named vars.
    """
    ns = dict(_EXEC_GLOBALS)
    ns.update(named_vars)
    exec(code, ns)  # noqa: S102
    return ns["get_constraints"](**named_vars)


# ── utilities ─────────────────────────────────────────────────────────────────

def _flatten(constraints: list) -> list:
    """Recursively flatten nested lists so Z3 never receives a list as an arg."""
    out = []
    for c in constraints:
        if isinstance(c, list):
            out.extend(_flatten(c))
        else:
            out.append(c)
    return out


def _add_domain_axioms(solver: Solver, named_vars: dict) -> None:
    """Add per-type bounds so the solver can produce sane witnesses."""
    for var in named_vars.values():
        if hasattr(var, "axioms"):
            solver.add(*var.axioms())
    # Pairwise dtype compatibility: prevent Z3 from pairing uint16/32/64 or
    # Float8 types across tensors.  promote_types throws for those cross-type
    # combos, so any generated input would fail at runtime for the wrong reason.
    from mobile.engine.model import compatible_dtypes, TensorVar
    dtypes = [var.dtype for var in named_vars.values() if isinstance(var, TensorVar)]
    for i in range(len(dtypes)):
        for j in range(i + 1, len(dtypes)):
            solver.add(compatible_dtypes(dtypes[i], dtypes[j]))


# ── coarse-grained detection ──────────────────────────────────────────────────

# Names of coarse-grained free Bools that block runtime verification.
_COARSE_GRAINED_NAMES = frozenset({
    "check_memory_format", "check_resizable", "check_scaling",
    "check_overlap", "check_internal_overlap",
    "check_name", "check_contiguous",
    "check_deterministic", "check_deterministic_fill",
    "check_sdp_backend",
})


def uses_coarse_grained(constraints: list) -> bool:
    """
    Return True if any coarse-grained free Bool appears anywhere in the
    constraint list.  Walks the Z3 expression DAG; visits each node at most once.
    """
    seen: set[int] = set()

    def _walk(expr) -> bool:
        eid = expr.get_id()
        if eid in seen:
            return False
        seen.add(eid)
        if is_const(expr) and expr.decl().name() in _COARSE_GRAINED_NAMES:
            return True
        return any(_walk(child) for child in expr.children())

    return any(_walk(c) for c in _flatten(constraints) if hasattr(c, "children"))


# ── Level 1: Z3 satisfiability ───────────────────────────────────────────────

def z3_check(constraints: list, named_vars: dict) -> str | None:
    """
    Check that the constraints are satisfiable given basic domain axioms.

    Returns None on success (constraints are consistent).
    Returns an error string if the constraints are unsatisfiable (LLM logic error).
    Empty constraint lists pass immediately.
    """
    constraints = _flatten(constraints)
    if not constraints:
        return None

    solver = _make_solver()
    _add_domain_axioms(solver, named_vars)
    solver.add(And(*constraints))

    result = solver.check()
    if result == sat:
        return None
    if result == unknown:
        return (
            f"Z3 timed out after {Z3_TIMEOUT_MS // 1000}s checking your constraints. "
            "This usually means the constraints involve non-linear arithmetic that Z3 "
            "cannot decide quickly (e.g. numel() products, large ITE chains, or "
            "is_contiguous() stride formulas).\n"
            "Simplify: replace numel() == 1 with numel_is_one(), replace "
            "is_contiguous() checks with check_contiguous, and avoid deeply-nested "
            "If/ITE expressions."
        )

    return (
        "Your constraints are unsatisfiable — no input can satisfy all conditions "
        "simultaneously. This usually means a logical contradiction (e.g. requiring "
        "dtype == INT32 AND dtype == FLOAT32 at the same time).\n"
        "Re-read the IR branches and fix the constraint logic."
    )


# ── Level 2: PyTorch runtime check ───────────────────────────────────────────

# RuntimeError messages that mark an unconstrainable value-edge (counted as SKIP,
# not FAIL) — see _call_op_forked.  Deliberately specific to avoid masking real bugs.
_SKIP_RUNTIME_PATTERNS = (
    "ZeroDivisionError",                  # integer div/mod by zero (floor_divide/fmod/remainder)
    "integer multiplication overflow",   # numel overflow from negative/huge int[] sizes
)


# Fatal signals that mean the op hit a HARD crash (segfault / abort / div-by-zero
# FPE).  Per the project rule these are "valid" (inconclusive, not a clean reject):
# the positive harness SKIPs them; the over-constraint checker counts them as
# witnesses.  SIGKILL (9) is excluded — that is the timeout kill WE send, not a crash.
HARD_CRASH_SIGNALS = frozenset({
    signal.SIGSEGV,   # 11 — segfault
    signal.SIGABRT,   #  6 — abort()/INTERNAL_ASSERT-that-aborts
    signal.SIGFPE,    #  8 — integer div-by-zero / floating-point exception
    signal.SIGBUS,    #  7 — bus error
    signal.SIGILL,    #  4 — illegal instruction
})


def _run_forked(op, args: list, timeout: float = 10.0) -> dict:
    """
    Call op(*args) in a forked child and report a rich outcome.

    Isolates SIGFPE / SIGSEGV and other crashes from the main process.  Uses
    os.fork() so the child inherits the already-imported torch module.  The child
    writes a pickled (tag, value) pair to a pipe and exits cleanly; a crash kills
    it before it writes, leaving the exit status' termination signal.

    Returns a dict:
      tag          : "ok" | "runtime" | "exc" | None   (None when no/garbled payload)
      val          : payload message (or None / "malformed result")
      have_payload : bool — the child wrote a (parseable or not) payload
      signaled     : bool — child died from a signal
      termsig      : int | None — the signal number (when signaled)
      timed_out    : bool — deadline elapsed; the child was SIGKILLed by us
    """
    r_fd, w_fd = os.pipe()
    pid = os.fork()

    if pid == 0:
        # ── child ──────────────────────────────────────────────────────
        os.close(r_fd)
        try:
            op(*args)
            payload = pickle.dumps(("ok", None))
        except (RuntimeError, IndexError, ValueError) as exc:
            payload = pickle.dumps(("runtime", str(exc)))
        except Exception as exc:
            payload = pickle.dumps(("exc", f"{type(exc).__name__}: {exc}"))
        try:
            os.write(w_fd, payload)
        except Exception:
            pass
        os.close(w_fd)
        os._exit(0)

    # ── parent ─────────────────────────────────────────────────────────
    os.close(w_fd)
    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout
    status: int | None = None
    reaped = False
    timed_out = False

    while True:
        remaining = max(0.0, deadline - time.monotonic())
        ready, _, _ = select.select([r_fd], [], [], min(remaining, 0.2))
        if ready:
            chunk = os.read(r_fd, 65536)
            if chunk:
                chunks.append(chunk)
                continue
            break  # EOF — child closed the write end (clean exit or crash)
        wpid, st = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            status, reaped = st, True
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break

    os.close(r_fd)
    if not reaped:
        # Reap without killing first so a crash's original termsig is preserved
        # (a dead/zombie child ignores our later SIGKILL).
        try:
            wpid, st = os.waitpid(pid, os.WNOHANG)
            if wpid == pid:
                status, reaped = st, True
        except ChildProcessError:
            reaped = True
        if not reaped:
            # Still running → timeout/hang.  Kill it (status will be SIGKILL=9,
            # which HARD_CRASH_SIGNALS excludes, so it reads as timeout not crash).
            timed_out = True
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                _, status = os.waitpid(pid, 0)
            except ChildProcessError:
                status = None

    signaled = status is not None and os.WIFSIGNALED(status)
    termsig = os.WTERMSIG(status) if signaled else None

    tag = val = None
    if chunks:
        try:
            tag, val = pickle.loads(b"".join(chunks))
        except Exception:
            tag, val = None, "malformed result"

    return {"tag": tag, "val": val, "have_payload": bool(chunks),
            "signaled": signaled, "termsig": termsig, "timed_out": timed_out}


def is_hard_crash(r: dict) -> bool:
    """True iff the forked child died from a hard-crash signal (not our timeout kill)."""
    return (not r["have_payload"]) and r["signaled"] and r["termsig"] in HARD_CRASH_SIGNALS


def _call_op_forked(op, args: list, timeout: float = 10.0) -> tuple[bool | None, str | None]:
    """
    Behavior-preserving wrapper over _run_forked used by the positive harness.

    Returns (triggered: bool | None, error: str | None):
      triggered=False → ran OK (PASS); =True → clean RuntimeError (FAIL);
      =None → inconclusive (crash/timeout/allocator/value-edge → SKIP).
    """
    r = _run_forked(op, args, timeout)

    if not r["have_payload"]:
        return None, "operator call inconclusive: subprocess crash or timeout"

    tag, val = r["tag"], r["val"]
    if tag is None:  # garbled payload
        return None, "operator call inconclusive: malformed result"
    if tag == "ok":
        return False, None
    if tag == "runtime":
        if any(s in val for s in ("can't allocate", "DefaultCPUAllocator",
                                  "out of memory", "alloc_cpu", "CUDA out of memory")):
            return None, f"operator call inconclusive: allocator error: {val[:80]}"
        # Unconstrainable value-edge errors → SKIP (not FAIL).  These are genuine
        # invalid inputs that no shape/dtype constraint can exclude and that we've
        # deliberately decided not to constrain:
        #   • integer division/modulo by zero (floor_divide/fmod/remainder with a
        #     zero in an integer divisor) → RuntimeError "ZeroDivisionError"
        #   • numel multiplication overflow from negative/huge int[] sizes
        if any(s in val for s in _SKIP_RUNTIME_PATTERNS):
            return None, f"operator call inconclusive (unconstrainable value edge): {val[:80]}"
        return True, val
    return None, f"operator call inconclusive: {val[:80]}"


def runtime_check(
    constraints: list,
    named_vars: dict,
    entry_sym: str,
    named_params: list[tuple[str, str]],
) -> dict:
    """
    Run Z3 to find a concrete bad input, then call the ATen op to confirm the
    TORCH_CHECK fires.

    Returns a dict:
      sat        : bool | None  — True if Z3 found a witness
      triggered  : bool | None  — True/False/None (None = inconclusive)
      witness    : dict | None  — {arg_name: concrete_value}
      error      : str | None
    """
    constraints = _flatten(constraints)
    if not constraints:
        return {"sat": None, "triggered": None, "witness": None,
                "error": "empty constraints (no conditions to check)"}

    solver = _make_solver()
    _add_domain_axioms(solver, named_vars)
    solver.add(And(*constraints))

    result = solver.check()
    if result == unknown:
        return {"sat": None, "triggered": None, "witness": None,
                "error": f"Z3 timeout ({Z3_TIMEOUT_MS // 1000}s) during witness search"}
    if result != sat:
        return {"sat": False, "triggered": None, "witness": None,
                "error": "constraints unsat"}

    model = solver.model()

    op, _, _ = find_op_for_symbol(entry_sym)
    concrete = build_call_args(named_params, named_vars, model)

    triggered = None
    error = "; ".join(concrete.errors) if concrete.errors else None
    if op is None:
        error = f"operator not found for symbol: {entry_sym[:60]}"
    elif not concrete.errors:
        triggered, error = _call_op_forked(op, concrete.args)

    return {
        "sat": True,
        "triggered": triggered,
        "witness": concrete.witness,
        "error": error,
    }


def _runtime_fail_msg(witness: dict) -> str:
    lines = [
        "Z3 found a concrete input satisfying your constraints, but calling the "
        "operator did NOT raise a RuntimeError.",
        "Your constraints may be too broad (allowing valid inputs) or encode the "
        "wrong conditions.",
        "",
        "Concrete witness (inputs your constraints label as 'bad'):",
    ]
    for name, val in witness.items():
        if val is not None and hasattr(val, "shape"):
            try:
                strides = list(val.stride())
            except Exception:
                strides = "<no strides>"
            lines.append(
                f"  {name}: ndim={val.ndim}, dtype={val.dtype}, "
                f"sizes={list(val.shape)}, strides={strides}"
            )
        else:
            lines.append(f"  {name}: {val!r}")
    lines += ["", "Re-read the IR and tighten your constraints."]
    return "\n".join(lines)


# ── orchestration ─────────────────────────────────────────────────────────────

def verify_constraints(
    constraints: list,
    named_vars: dict,
    entry_sym: str,
    named_params: list[tuple[str, str]],
) -> tuple[str | None, bool]:
    """
    Run Z3 satisfiability check then PyTorch runtime check on already-evaluated
    constraints.

    Parameters
    ----------
    constraints  : list of Z3 expressions returned by get_constraints()
    named_vars   : symbolic variable dict built by build_vars()
    entry_sym    : mangled C++ symbol for the ATen operator
    named_params : [(name, pytorch_type), ...] for concretization

    Returns (error, inconclusive):
      error        — error string for the LLM to fix, or None on success.
      inconclusive — True when Z3 timed out during witness search (triggered=None);
                     caller should tag the file '# verified: inconclusive'.
    """
    # Level 1: Z3 sat check.
    try:
        z3_err = z3_check(constraints, named_vars)
    except Exception as exc:
        return (
            f"Your constraints raised a Z3 error: {exc}\n"
            "Common causes: passing a method without calling it "
            "(e.g. arg1.is_floating_point instead of arg1.is_floating_point()), "
            "or putting a non-Z3 object in the returned list.\n"
            "Fix get_constraints() and try again.",
            False,
        )
    if z3_err is not None:
        return z3_err, False

    if not constraints:
        return None, False

    try:
        result = runtime_check(constraints, named_vars, entry_sym, named_params)
    except Exception as exc:
        return (
            f"Your constraints raised a Z3 error during runtime check: {exc}\n"
            "Fix get_constraints() and try again.",
            False,
        )
    if result["triggered"] is False:
        return _runtime_fail_msg(result["witness"]), False

    # triggered=True → fully verified; triggered=None → Z3 timeout (inconclusive).
    inconclusive = result["triggered"] is None
    return None, inconclusive


# ── file annotation ───────────────────────────────────────────────────────────

def _inject_tag(code: str, tag: str) -> str:
    """Insert '# verified: <tag>' on the line after '# description:'."""
    lines = code.splitlines()
    out = []
    for line in lines:
        out.append(line)
        stripped = line.lstrip()
        if stripped.startswith("# description:") and "# verified:" not in code:
            indent = " " * (len(line) - len(stripped))
            out.append(f"{indent}# verified: {tag}")
    return "\n".join(out)


def _inject_verified_tag(code: str) -> str:
    return _inject_tag(code, "runtime")


def _inject_inconclusive_tag(code: str) -> str:
    return _inject_tag(code, "inconclusive")


def maybe_mark_verified(code: str, constraints: list) -> str:
    """Annotate with '# verified: runtime' when runtime check confirmed the error fires."""
    if constraints:
        return _inject_verified_tag(code)
    return code


def maybe_mark_inconclusive(code: str, constraints: list) -> str:
    """Annotate with '# verified: inconclusive' when Z3 timed out during witness search."""
    if constraints:
        return _inject_inconclusive_tag(code)
    return code


def _mark_verified(py_path: Path, code: str) -> None:
    """Rewrite py_path with '# verified: runtime' tag if not already present."""
    if "# verified: runtime" in code:
        return
    py_path.write_text(_inject_verified_tag(code))


# ── standalone CLI ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a single constraint file against a .ll slice"
    )
    parser.add_argument("constraint_py", help="Path to the constraint .py file")
    parser.add_argument("ll_file",       help="Path to the .ll slice file")
    opts = parser.parse_args()

    from ir_reader import read_slice  # local import to avoid circular dependency

    constraint_path = Path(opts.constraint_py)
    ll_path = Path(opts.ll_file)

    if not constraint_path.exists():
        print(f"ERROR: constraint file not found: {constraint_path}", file=sys.stderr)
        return 1
    if not ll_path.exists():
        print(f"ERROR: .ll file not found: {ll_path}", file=sys.stderr)
        return 1

    # Read the slice info
    slice_info = read_slice(ll_path)
    if slice_info is None:
        print(f"ERROR: could not parse .ll file: {ll_path}", file=sys.stderr)
        return 1

    print(f"Slice     : {slice_info.slice_name}", flush=True)
    print(f"Symbol    : {slice_info.entry_sym[:80]}", flush=True)
    print(f"Demangled : {slice_info.entry_demangled[:100]}", flush=True)
    print(f"Check     : {slice_info.check_file}:{slice_info.check_line}", flush=True)
    print(f"In loop   : {slice_info.in_loop}", flush=True)
    print(f"Params    : {slice_info.named_params}", flush=True)

    # Build symbolic variables
    named_vars = build_vars(slice_info.named_params)

    # Read and exec constraint code
    code = constraint_path.read_text()
    print(f"\nExec-validating {constraint_path.name} ...", flush=True)
    try:
        constraints = exec_constraints(code, named_vars)
    except Exception:
        print(f"EXEC ERROR:\n{traceback.format_exc(limit=5)}", flush=True)
        return 1

    print(f"  Constraints: {len(constraints)} expression(s)", flush=True)

    if not constraints:
        print("  RESULT: EMPTY (no constraints)", flush=True)
        return 0

    # Z3 check
    print("\nZ3 satisfiability check ...", flush=True)
    z3_err = z3_check(constraints, named_vars)
    if z3_err is not None:
        print(f"  Z3 FAIL: {z3_err[:120]}", flush=True)
        return 1
    print("  Z3: SAT (consistent)", flush=True)

    # Runtime check
    print("\nRuntime check ...", flush=True)
    result = runtime_check(constraints, named_vars, slice_info.entry_sym, slice_info.named_params)
    if result["sat"] is None:
        print(f"  INCONCLUSIVE: {result['error']}", flush=True)
        return 0
    if not result["sat"]:
        print("  UNSAT: constraints are unsatisfiable", flush=True)
        return 1
    if result["triggered"] is True:
        print("  TRIGGERED: RuntimeError fired — constraints verified!", flush=True)
        return 0
    if result["triggered"] is False:
        print("  NOT TRIGGERED: operator did not raise RuntimeError", flush=True)
        if result.get("witness"):
            print("  Witness:", flush=True)
            for k, v in result["witness"].items():
                if v is not None and hasattr(v, "shape"):
                    print(f"    {k}: ndim={v.ndim} dtype={v.dtype} shape={list(v.shape)}", flush=True)
                else:
                    print(f"    {k}: {v!r}", flush=True)
        return 1
    print(f"  INCONCLUSIVE: {result.get('error', 'unknown')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
