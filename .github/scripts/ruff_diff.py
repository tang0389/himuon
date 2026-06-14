#!/usr/bin/env python3
"""Diff-aware ruff — lint only the lines you actually changed.

Existing code is grandfathered: a pre-existing violation in a file you touch
does NOT fail the check. Only ``ruff check`` findings on lines you added or
modified are reported. Formatting is verified on **newly-added files only**,
because ``ruff format`` rewrites whole files and so cannot enforce line-level
formatting without reformatting (and thus touching) legacy code.

Diff source (pick one):
    --base <ref>   compare ``<ref>...HEAD``           (CI: PR / push)
    --staged       compare the staged index vs HEAD   (pre-commit)
    (default)      compare the working tree vs HEAD   (local convenience)

Exit status is 1 if any in-range lint finding or added-file format problem is
found, 0 otherwise, 2 on tooling error. Only standard library is used so the
script runs under any Python; ``ruff`` is invoked through ``uvx`` (no project
dependency required).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

# Pinned to match .pre-commit-config.yaml so CI, pre-commit, and local runs
# all agree on the ruleset.
RUFF_VERSION = "0.14.4"

# Parses a unified-diff hunk header, capturing the new-file (``+``) side:
#   @@ -<old_start>[,<old_len>] +<new_start>[,<new_len>] @@
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout


def _ruff(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["uvx", f"ruff@{RUFF_VERSION}", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _range_args(base: str | None, staged: bool) -> list[str]:
    if base:
        return [f"{base}...HEAD"]  # three-dot: merge-base(base, HEAD)..HEAD
    if staged:
        return ["--cached"]
    return ["HEAD"]


def _changed_py_files(range_args: list[str], diff_filter: str = "ACM") -> list[str]:
    out = _git("diff", "--name-only", f"--diff-filter={diff_filter}", *range_args, "--", "*.py")
    return [f for f in out.splitlines() if f]


def _changed_lines(path: str, range_args: list[str]) -> set[int]:
    """1-based line numbers added/modified in ``path`` on the new side."""
    out = _git("diff", "--unified=0", *range_args, "--", path)
    lines: set[int] = set()
    for line in out.splitlines():
        m = _HUNK_RE.match(line)
        if not m:
            continue
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) is not None else 1
        lines.update(range(start, start + count))
    return lines


def _lint(changed: dict[str, set[int]]) -> list[str]:
    files = list(changed)
    if not files:
        return []
    proc = _ruff("check", "--output-format=json", "--force-exclude", *files)
    if proc.returncode not in (0, 1):  # 0 = clean, 1 = findings; else tooling error
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
        sys.exit(2)
    try:
        findings = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        sys.stderr.write("Could not parse ruff JSON output:\n" + proc.stdout + "\n")
        sys.exit(2)

    msgs: list[str] = []
    for f in findings:
        path = os.path.relpath(f["filename"])
        loc = f.get("location") or {}
        row = loc.get("row")
        if row is None or row not in changed.get(path, ()):
            continue
        code = f.get("code") or "?"
        col = loc.get("column", 0)
        msgs.append(f"{path}:{row}:{col}: {code} {f.get('message', '')}".rstrip())
    return sorted(msgs)


def _format_added(range_args: list[str]) -> list[str]:
    added = _changed_py_files(range_args, diff_filter="A")
    if not added:
        return []
    proc = _ruff("format", "--check", *added)
    if proc.returncode == 0:
        return []
    return [ln for ln in proc.stdout.splitlines() if ln] or [
        "newly-added files are not formatted; run: uvx ruff format <files>"
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--base", help="compare <base>...HEAD (CI)")
    g.add_argument(
        "--staged", action="store_true", help="compare staged index vs HEAD (pre-commit)"
    )
    args = ap.parse_args()

    range_args = _range_args(args.base, args.staged)
    files = _changed_py_files(range_args)
    if not files:
        print("No changed Python files; nothing to lint.")
        return 0

    changed = {f: _changed_lines(f, range_args) for f in files}
    lint_msgs = _lint(changed)
    fmt_msgs = _format_added(range_args)

    if lint_msgs:
        print("Lint findings on changed lines:")
        print("\n".join("  " + m for m in lint_msgs))
    if fmt_msgs:
        print("Formatting issues in newly-added files:")
        print("\n".join("  " + m for m in fmt_msgs))
    if lint_msgs or fmt_msgs:
        return 1

    print(f"Checked {len(files)} changed file(s); no findings on changed lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
