#!/usr/bin/env python3
"""build_skips.py — produce the per-operator SKIP reason table (step 4/6 of analysis_single.md),
splitting OpenVINO delegate blob failures (error 0x1, CALL_DELEGATE) from portable-kernel
Check-failed rejections (error 0x12), plus the CRASH table. Writes skips.md.
"""
from __future__ import annotations
import re, json, collections
from pathlib import Path

ROOT = Path("/data/jwen929/mobile")
OUT = ROOT / "findings/openvino_single"
SKIPLOG = OUT / "skiplog.tsv"
MANIFEST = OUT / "manifest.tsv"

OP_RE = re.compile(r"n0\*?=([A-Za-z0-9_.]+?)\(")


def op_of(graph):
    m = OP_RE.search(graph or "")
    return m.group(1) if m else "?"


def deleg_of():
    d = {}
    with open(MANIFEST) as f:
        next(f, None)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                d[p[0]] = (int(p[2]) if p[2] != "" else None)
    return d


def skip_signature(reason: str) -> str:
    """Collapse a runtime reason to a stable signature."""
    r = reason
    # OpenVINO delegate execute failure
    if "CALL_DELEGATE" in r:
        return "OV-delegate: CALL_DELEGATE execute failed (blob won't run)"
    # portable kernel Check failed — keep the [file.cpp:NN] Check failed (...) clause
    m = re.search(r"\[([a-zA-Z0-9_.]+\.(?:cpp|h)):\d+\]\s*(Check failed \([^)]*\)|Unhandled dtype \w+|[^|]+?not supported[^|]*)", r)
    if m:
        return f"portable {m.group(1)}: {m.group(2).strip()}"
    m2 = re.search(r"\[([a-zA-Z0-9_.]+\.(?:cpp|h)):\d+\]", r)
    return f"portable {m2.group(1)}" if m2 else r[:80]


def main():
    deleg = deleg_of()
    skip_rows = []
    crash_rows = []
    with open(SKIPLOG) as f:
        next(f, None)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                p += [""] * (4 - len(p))
            status, job, reason, graph = p[0], p[1], p[2], p[3]
            op = op_of(graph)
            if status == "SKIP":
                skip_rows.append((op, job, reason))
            elif status == "CRASH":
                crash_rows.append((op, job, reason))

    # SKIP: split OV-delegate vs portable, count per (op, signature)
    ov_sig = collections.Counter()
    port_sig = collections.Counter()
    ov_ops = collections.Counter()
    port_ops = collections.Counter()
    for op, job, reason in skip_rows:
        sig = skip_signature(reason)
        if sig.startswith("OV-delegate"):
            ov_ops[op] += 1
        else:
            port_ops[op] += 1
            port_sig[(op, sig)] += 1

    # CRASH: split delegated vs portable
    crash_deleg = collections.Counter()
    crash_port = collections.Counter()
    for op, job, reason in crash_rows:
        d = deleg.get(job)
        if d is not None and d >= 1:
            crash_deleg[op] += 1
        else:
            crash_port[op] += 1

    lines = []
    lines.append("# OpenVINO single-op — SKIP & CRASH reason tables\n")
    lines.append("Every SKIP/CRASH is one operator (single-op corpus). The *reason* is the finding.\n")

    lines.append("\n## SKIP class A — OpenVINO delegate blob failure (coverage gap)\n")
    lines.append("Runtime reason (verbatim, all identical modulo the hex code):\n")
    lines.append("```\nclient RuntimeError: method->execute() failed with error 0x1 | "
                 "[OpenvinoBackend.cpp:98] OpenVINO: runtime loaded successfully ... | "
                 "[method.cpp:1525] CALL_DELEGATE execute failed at instruction 0: 0x1\n```\n")
    lines.append("The op partitions to OpenVINO and its blob compiles, but the compiled model "
                 "**fails at execute** — a real OpenVINO coverage gap. Ops (count desc):\n")
    lines.append("| count | operator |\n|--:|---|")
    for op, n in ov_ops.most_common():
        lines.append(f"| {n} | `{op}` |")
    lines.append(f"\n**Total OV-delegate SKIPs: {sum(ov_ops.values())} across {len(ov_ops)} operators.**\n")

    lines.append("\n## SKIP class B — portable-kernel rejection (error 0x12, NOT OpenVINO)\n")
    lines.append("These ops fell back to portable CPU kernels which then raised a `Check failed` / "
                 "unsupported-dtype / unsupported-arg guard. Per step 3a these are **portable-side "
                 "coverage gaps, not OpenVINO bugs** — listed with the verbatim guard clause.\n")
    lines.append("| count | operator | portable guard (verbatim) |\n|--:|---|---|")
    for (op, sig), n in port_sig.most_common():
        lines.append(f"| {n} | `{op}` | {sig} |")
    lines.append(f"\n**Total portable-rejection SKIPs: {sum(port_ops.values())} across {len(port_ops)} operators.**\n")

    lines.append("\n## CRASH — native aborts\n")
    lines.append("All CRASHes logged `executor died (native abort)` — a native SIGABRT/SIGSEGV with "
                 "**no catchable error message**. The missing guard is the bug (the kernel should "
                 "reject the form with a catchable error, not abort).\n")
    lines.append("\n### CRASH on the OpenVINO delegate (ops>=1) — real finding\n")
    lines.append("| count | operator | trigger |\n|--:|---|---|")
    for op, n in crash_deleg.most_common():
        lines.append(f"| {n} | `{op}` | native abort (no catchable message) |")
    lines.append(f"\n**Total delegated CRASHes: {sum(crash_deleg.values())} across {len(crash_deleg)} operators.**\n")
    lines.append("\n### CRASH on portable fallback (ops=0) — ruled out (not OpenVINO)\n")
    lines.append("| count | operator |\n|--:|---|")
    for op, n in crash_port.most_common():
        lines.append(f"| {n} | `{op}` |")
    lines.append(f"\n**Total portable CRASHes: {sum(crash_port.values())} across {len(crash_port)} operators.**\n")

    (OUT / "skips.md").write_text("\n".join(lines))
    print(f"OV-delegate SKIP ops: {len(ov_ops)} ({sum(ov_ops.values())} jobs)")
    print(f"portable SKIP ops:    {len(port_ops)} ({sum(port_ops.values())} jobs)")
    print(f"delegated CRASH ops:  {len(crash_deleg)} ({sum(crash_deleg.values())} jobs): {dict(crash_deleg)}")
    print(f"portable CRASH ops:   {len(crash_port)} ({sum(crash_port.values())} jobs): {dict(crash_port)}")
    print("wrote skips.md")


if __name__ == "__main__":
    main()
