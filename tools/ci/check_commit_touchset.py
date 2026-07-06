#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Touch-set gate — a commit's file set must be a subset of the line's declared touch-set.

Parallel work lines (see docs/并行线纪律-worktree-与触碰集.md) each declare, up front,
the set of path globs they are allowed to write. This script is the machine check:

    given a commit (or the staged set), verify

        { files touched } ⊆ { globs declared by the active line }

    out-of-bounds → exit non-zero (red).

Touch-set declaration (two sources, unioned):
  1. `.touch-set/<line>.txt`  — one path-glob per line, `#` starts a comment.
                                 This is the durable, git-tracked declaration.
  2. commit-trailer           — `Touch-Set: <glob> <glob> ...` in the commit message
                                 (ad-hoc / override; unioned with the line file).
                                 `Touch-Set-Line: <name>` names the active line for
                                 a commit when no other source is given.

Active-line resolution order (first hit wins):
  --line NAME  >  $TRELLIS_TOUCH_SET_LINE  >  `Touch-Set-Line:` trailer (commit mode)
  >  `.touch-set/ACTIVE` marker file (per-worktree, git-ignored)

Glob semantics (predictable, path-aware):
  foo/bar.py     exact file
  tools/ci/      trailing slash → directory prefix (everything under it)
  tools/ci/*.py  `*`  matches within one path segment (not `/`)
  .trellis/**    `**` matches across `/` (recursive subtree)
  ?              single non-`/` char

Stdlib-only (matches .github/workflows/falsifier-gate.yml conventions).

CLI:
  check_commit_touchset.py --staged            # gate the staged set (pre-commit)
  check_commit_touchset.py --commit <ref>      # gate a single commit
  check_commit_touchset.py --range A..B        # gate every commit in a range
  check_commit_touchset.py --line <name> ...   # force the active line
  check_commit_touchset.py --self-test         # hermetic + real-git self-test

Exit codes: 0 = all files in bounds · 2 = violation · 3 = misconfiguration.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

VIOLATION_EXIT = 2
CONFIG_EXIT = 3

TRAILER_LINE_RE = re.compile(r"^Touch-Set-Line:\s*(?P<name>\S+)\s*$")
TRAILER_GLOBS_RE = re.compile(r"^Touch-Set:\s*(?P<globs>.+?)\s*$")


# ─── git helpers ──────────────────────────────────────────────────────────────

def _git(args: Sequence[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def repo_root(cwd: Optional[Path] = None) -> Path:
    rc, out, _ = _git(["rev-parse", "--show-toplevel"], cwd=cwd)
    if rc != 0:
        return Path(cwd or os.getcwd()).resolve()
    return Path(out.strip()).resolve()


def commit_files(ref: str, root: Path) -> List[str]:
    rc, out, err = _git(
        ["diff-tree", "--no-commit-id", "--name-only", "-r", ref], cwd=root
    )
    if rc != 0:
        raise RuntimeError(f"git diff-tree failed for {ref}: {err.strip()}")
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def staged_files(root: Path) -> List[str]:
    rc, out, err = _git(["diff", "--cached", "--name-only"], cwd=root)
    if rc != 0:
        raise RuntimeError(f"git diff --cached failed: {err.strip()}")
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def range_commits(rng: str, root: Path) -> List[str]:
    rc, out, err = _git(["rev-list", "--reverse", rng], cwd=root)
    if rc != 0:
        raise RuntimeError(f"git rev-list failed for {rng}: {err.strip()}")
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def commit_message(ref: str, root: Path) -> str:
    rc, out, _ = _git(["show", "-s", "--format=%B", ref], cwd=root)
    return out if rc == 0 else ""


# ─── touch-set resolution ─────────────────────────────────────────────────────

def parse_trailers(message: str) -> Tuple[Optional[str], List[str]]:
    """Return (line-name-or-None, inline-globs) parsed from commit-message trailers."""
    line_name: Optional[str] = None
    globs: List[str] = []
    for raw in message.splitlines():
        m = TRAILER_LINE_RE.match(raw.strip())
        if m:
            line_name = m.group("name")
            continue
        m = TRAILER_GLOBS_RE.match(raw.strip())
        if m:
            globs.extend(m.group("globs").split())
    return line_name, globs


def read_line_file(line: str, root: Path) -> List[str]:
    path = root / ".touch-set" / f"{line}.txt"
    if not path.is_file():
        return []
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def resolve_line(
    explicit: Optional[str], root: Path, trailer_line: Optional[str]
) -> Optional[str]:
    if explicit:
        return explicit
    env = os.environ.get("TRELLIS_TOUCH_SET_LINE")
    if env:
        return env.strip()
    if trailer_line:
        return trailer_line
    marker = root / ".touch-set" / "ACTIVE"
    if marker.is_file():
        val = marker.read_text(encoding="utf-8").strip()
        if val:
            return val
    return None


# ─── glob matching ────────────────────────────────────────────────────────────

def _glob_to_regex(pattern: str) -> re.Pattern:
    """Path-aware glob → regex. `*`=segment, `**`=recursive, `?`=one non-slash char."""
    i = 0
    out = ["^"]
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                out.append(".*")
                i += 2
                # swallow an immediately following slash so `a/**/b` and `a/**`
                # both behave, and `a/**` matches `a/` prefix cleanly.
                if i < n and pattern[i] == "/":
                    i += 1
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(c))
        i += 1
    out.append("$")
    return re.compile("".join(out))


def _strip_dot_slash(p: str) -> str:
    p = p.strip()
    while p.startswith("./"):
        p = p[2:]
    return p


def path_matches(path: str, pattern: str) -> bool:
    path = _strip_dot_slash(path)
    pattern = _strip_dot_slash(pattern)
    if not pattern:
        return False
    # Directory-prefix form: trailing slash, or a bare path with no wildcard that
    # names a directory (we cannot stat in a bare-list context, so treat a
    # wildcard-free pattern also as a prefix match to be permissive on dirs).
    if pattern.endswith("/"):
        prefix = pattern
        return path == prefix[:-1] or path.startswith(prefix)
    if not any(ch in pattern for ch in "*?"):
        # exact file OR directory prefix (dir declared without trailing slash)
        return path == pattern or path.startswith(pattern.rstrip("/") + "/")
    return _glob_to_regex(pattern).match(path) is not None


def in_touchset(path: str, globs: Sequence[str]) -> bool:
    return any(path_matches(path, g) for g in globs)


# ─── core check ───────────────────────────────────────────────────────────────

def check_fileset(
    files: Sequence[str], globs: Sequence[str]
) -> List[str]:
    """Return the list of files that are OUT of the touch-set (empty ⇒ all in bounds)."""
    return [f for f in files if not in_touchset(f, globs)]


def _report(label: str, files: Sequence[str], globs: Sequence[str], line: str) -> int:
    violations = check_fileset(files, globs)
    if violations:
        print(f"[touch-set] RED  {label}: {len(violations)} file(s) outside line '{line}':")
        for f in violations:
            print(f"    ✗ {f}")
        print(f"[touch-set] declared globs for '{line}': {', '.join(globs) or '(none)'}")
        print("[touch-set] fix: stage only declared paths (no `git add -A`), or")
        print(f"           widen `.touch-set/{line}.txt` on purpose, then retry.")
        return VIOLATION_EXIT
    print(f"[touch-set] GREEN {label}: {len(files)} file(s) ⊆ line '{line}'.")
    return 0


def run_check(args: argparse.Namespace) -> int:
    root = repo_root()

    # commit / range modes may carry a trailer that names the line
    trailer_line: Optional[str] = None
    trailer_globs: List[str] = []

    if args.commit:
        msg = commit_message(args.commit, root)
        trailer_line, trailer_globs = parse_trailers(msg)

    line = resolve_line(args.line, root, trailer_line)
    if not line and not trailer_globs:
        print(
            "[touch-set] no active line declared "
            "(--line / $TRELLIS_TOUCH_SET_LINE / Touch-Set-Line: / .touch-set/ACTIVE) "
            "and no inline Touch-Set: trailer — cannot gate.",
            file=sys.stderr,
        )
        return CONFIG_EXIT

    globs = list(trailer_globs)
    if line:
        globs.extend(read_line_file(line, root))
    if not globs:
        print(
            f"[touch-set] line '{line}' declares no globs "
            f"(missing/empty .touch-set/{line}.txt and no inline trailer).",
            file=sys.stderr,
        )
        return CONFIG_EXIT

    line_label = line or "(inline-trailer)"

    if args.staged:
        return _report("staged", staged_files(root), globs, line_label)

    if args.commit:
        return _report(f"commit {args.commit}", commit_files(args.commit, root), globs, line_label)

    if args.range:
        worst = 0
        for ref in range_commits(args.range, root):
            # per-commit trailer can re-point the line inside a range
            msg = commit_message(ref, root)
            c_line, c_globs = parse_trailers(msg)
            eff_line = resolve_line(args.line, root, c_line) or line
            eff_globs = list(c_globs)
            if eff_line:
                eff_globs.extend(read_line_file(eff_line, root))
            if not eff_globs:
                eff_globs = globs
            rc = _report(f"commit {ref[:12]}", commit_files(ref, root), eff_globs, eff_line or line_label)
            worst = max(worst, rc)
        return worst

    print("[touch-set] nothing to check: pass --staged, --commit REF, or --range A..B", file=sys.stderr)
    return CONFIG_EXIT


# ─── self-test ────────────────────────────────────────────────────────────────

def _selftest_units() -> List[str]:
    """Hermetic path_matches / check_fileset assertions. Return failure messages."""
    fails: List[str] = []

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    # directory-prefix
    expect(path_matches("tools/ci/x.py", "tools/ci/"), "trailing-slash prefix")
    expect(path_matches("tools/ci/x.py", "tools/ci"), "bare-dir prefix")
    expect(not path_matches("tools/cix.py", "tools/ci/"), "prefix must respect boundary")
    # single-segment star
    expect(path_matches("tools/ci/x.py", "tools/ci/*.py"), "segment star match")
    expect(not path_matches("tools/ci/sub/x.py", "tools/ci/*.py"), "star must not cross /")
    # recursive
    expect(path_matches("tools/ci/sub/x.py", "tools/ci/**"), "double-star recursive")
    expect(path_matches("a/b/c/d", "a/**/d"), "double-star mid recursive")
    # exact
    expect(path_matches("docs/policy.md", "docs/policy.md"), "exact match")
    expect(not path_matches("docs/policy.md", "docs/other.md"), "exact mismatch")
    # dotfile-prefixed dir must not be mangled by ./ stripping
    expect(path_matches(".touch-set/ACTIVE", ".touch-set/"), "dotdir prefix match")
    expect(path_matches("./docs/x.md", "docs/"), "leading ./ stripped once")
    # subset check
    expect(check_fileset(["tools/ci/a.py"], ["tools/ci/"]) == [], "subset all-in")
    expect(check_fileset(["lib/x.cpp"], ["tools/ci/"]) == ["lib/x.cpp"], "subset violation")
    return fails


def _selftest_git() -> List[str]:
    """Real git: a compliant commit is GREEN, an out-of-bounds commit is RED."""
    fails: List[str] = []
    script = Path(__file__).resolve()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _git(["init", "-q"], cwd=root)
        _git(["config", "user.email", "t@t"], cwd=root)
        _git(["config", "user.name", "t"], cwd=root)
        (root / ".touch-set").mkdir()
        (root / ".touch-set" / "line-x.txt").write_text(
            "# line-x may only touch tools/ci and docs\ntools/ci/\ndocs/\n",
            encoding="utf-8",
        )
        (root / "tools" / "ci").mkdir(parents=True)
        (root / "docs").mkdir()
        _git(["add", ".touch-set"], cwd=root)
        _git(["commit", "-q", "-m", "seed touch-set"], cwd=root)

        # compliant commit
        (root / "tools" / "ci" / "ok.py").write_text("x=1\n", encoding="utf-8")
        (root / "docs" / "note.md").write_text("ok\n", encoding="utf-8")
        _git(["add", "tools/ci/ok.py", "docs/note.md"], cwd=root)
        _git(["commit", "-q", "-m", "compliant"], cwd=root)
        rc_ok, _, _ = _git(
            ["rev-parse", "HEAD"], cwd=root
        )
        good = subprocess.run(
            [sys.executable, str(script), "--line", "line-x", "--commit", "HEAD"],
            cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if good.returncode != 0:
            fails.append(f"compliant commit should be GREEN (exit0), got {good.returncode}:\n{good.stdout}")

        # out-of-bounds commit (touches lib/)
        (root / "lib").mkdir()
        (root / "lib" / "x.cpp").write_text("int x;\n", encoding="utf-8")
        _git(["add", "lib/x.cpp"], cwd=root)
        _git(["commit", "-q", "-m", "out of bounds"], cwd=root)
        bad = subprocess.run(
            [sys.executable, str(script), "--line", "line-x", "--commit", "HEAD"],
            cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if bad.returncode != VIOLATION_EXIT:
            fails.append(f"out-of-bounds commit should be RED (exit {VIOLATION_EXIT}), got {bad.returncode}:\n{bad.stdout}")

        # staged out-of-bounds (pre-commit path)
        (root / "lib" / "y.cpp").write_text("int y;\n", encoding="utf-8")
        _git(["add", "lib/y.cpp"], cwd=root)
        staged = subprocess.run(
            [sys.executable, str(script), "--line", "line-x", "--staged"],
            cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if staged.returncode != VIOLATION_EXIT:
            fails.append(f"staged out-of-bounds should be RED (exit {VIOLATION_EXIT}), got {staged.returncode}:\n{staged.stdout}")
    return fails


def run_selftest() -> int:
    fails = _selftest_units() + _selftest_git()
    if fails:
        print("[touch-set] SELF-TEST FAILED:")
        for f in fails:
            print(f"    ✗ {f}")
        return 1
    print("[touch-set] SELF-TEST PASSED "
          "(units + real-git compliant→GREEN / out-of-bounds→RED / staged→RED).")
    return 0


# ─── entry ────────────────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Touch-set gate: commit files ⊆ declared line globs.")
    ap.add_argument("--line", help="Force the active line name.")
    ap.add_argument("--staged", action="store_true", help="Check the staged set (pre-commit).")
    ap.add_argument("--commit", help="Check a single commit ref.")
    ap.add_argument("--range", help="Check every commit in a range (A..B).")
    ap.add_argument("--self-test", action="store_true", help="Run hermetic + real-git self-test.")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_selftest()
    try:
        return run_check(args)
    except RuntimeError as exc:
        print(f"[touch-set] error: {exc}", file=sys.stderr)
        return CONFIG_EXIT


if __name__ == "__main__":
    sys.exit(main())
