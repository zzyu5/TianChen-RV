#!/usr/bin/env python3
"""rename_to_weft.py — mechanical, byte-exact-preserving whole-project rename → Weft.

[RENAME] 裁四 (2026-07-12). Renames the project identifier family from the
TianChen-RV / TCRV lineage to Weft, across the FUNCTIONAL + CI + living-contract
surface. Historical append-only records (.trellis/backup, .trellis/tasks,
.trellis/workspace, docs/reports, artifacts/, experiments/, tools/e2e-harness,
tools/bench) are LEFT UNTOUCHED as named exceptions (the old name is factually
correct-at-the-time; rewriting corrupts the audit trail and does not affect
byte-exact / CI / build).

Token map (case-sensitive, mutually non-overlapping substrings, order-independent):

    TianChen-RV  -> Weft-RV     (hyphenated RISC-V-instance prose)
    TianChenRV   -> Weft        (CamelCase code identifier / project / CMake target)
    TIANCHENRV   -> WEFT        (ALL-CAPS CMake option/var)
    tianchenrv   -> weft        (C++ namespace)
    TCRV         -> WEFT        (acronym: .td def names, C++ classes, macros)
    tcrv         -> weft        (dialect prefix: tcrv_rvv, tcrv-opt, tcrvrvv, tcrv.)

DELIBERATELY NOT in the map: `TianchenRV` (capital-T, lowercase-c) — this is the
checkout directory basename (/home/kingdom/phdworks/TianchenRV) and appears only in
absolute filesystem paths + historical prose; renaming it would break real paths.
None of the six mapped tokens is a substring of `TianchenRV`, so it is preserved
automatically.

Byte-exact invariant: the SAME token substitution is applied to emitter string
literals (lib/) and to their goldens (test/**/*.inc, *.mlir), so re-emit stays
structurally byte-identical (only names change) and goldens stay matched.

Usage:
    python3 tools/rename_to_weft.py --plan        # print scope + move plan, change nothing
    python3 tools/rename_to_weft.py --content     # rewrite file contents in place
    python3 tools/rename_to_weft.py --moves       # git mv path renames
    python3 tools/rename_to_weft.py --all         # content then moves
    python3 tools/rename_to_weft.py --scope schema/   # restrict to a prefix (A-segment)
"""
import argparse
import os
import subprocess
import sys

REPO = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()

# Ordered. The FIRST rule is load-bearing: the C++ namespace was
# `tianchenrv::tcrv::<dialect>` (project::dialect-group::dialect). Mapping BOTH
# `tianchenrv`->weft and `tcrv`->weft independently would produce a doubled
# `weft::weft::<dialect>` whose inner `weft` shadows the project root and breaks
# cross-group lookup (`weft::plugin` resolves into `weft::weft`). So collapse the
# qualified pair `tianchenrv::tcrv` -> single `weft` FIRST; dialects then sit at
# `weft::rvv` etc., peers of `weft::plugin`/`weft::support` (idiomatic MLIR).
# The remaining rules are mutually non-overlapping case-sensitive substrings.
REPLACEMENTS = [
    ("tianchenrv::tcrv", "weft"),
    ("TianChen-RV", "Weft-RV"),
    ("TianChenRV", "Weft"),
    ("TIANCHENRV", "WEFT"),
    ("tianchenrv", "weft"),
    ("TCRV", "WEFT"),
    ("tcrv", "weft"),
]

import re

# A handful of hand-written TUs open the project+dialect-group namespace in BLOCK
# form (`namespace tianchenrv {` \n `namespace tcrv {`) rather than the C++17
# nested form (`namespace tianchenrv::tcrv::rvv {`). The compound rule above only
# collapses the nested/qualified forms; these two adjacent block openers/closers
# must be de-duplicated so the declaration matches the collapsed `weft::rvv`.
# Matches ONLY the doubled `weft` block (both segments exactly `weft`, no `::`),
# which is unique to the former tianchenrv/tcrv nesting.
_NS_OPEN_DUP = re.compile(r"namespace weft \{\n(\s*)namespace weft \{")
_NS_CLOSE_DUP = re.compile(r"\} // namespace weft\n(\s*)\} // namespace weft")


def collapse_block_namespace(s: str) -> str:
    s = _NS_OPEN_DUP.sub("namespace weft {", s)
    s = _NS_CLOSE_DUP.sub("} // namespace weft", s)
    return s

# Functional + CI + living-contract surface (git-tracked paths under these prefixes).
INCLUDE_PREFIXES = (
    "include/", "lib/", "schema/", "cmake/", "scripts/", "test/", ".github/",
    "tools/tcrv-opt/", "tools/tcrv-translate/",           # pre-move names
    "tools/weft-opt/", "tools/weft-translate/",           # post-move names
    "tools/lint/", "tools/fuzz/",
    "tools/ci/", "tools/visibility/", "tools/hooks/",
    ".trellis/scripts/", ".trellis/spec/",
    "docs/canon/", "docs/method/",
)
INCLUDE_FILES = (
    "CMakeLists.txt", "tools/CMakeLists.txt",
    "README.md", "CLAUDE.md", "AGENTS.md", "RTK.md",
)
# Named exceptions (historical / append-only) never touched:
#   .trellis/backup .trellis/tasks .trellis/workspace .trellis/.template-hashes.json
#   docs/reports docs/* (non canon/method) artifacts/ experiments/
#   tools/e2e-harness tools/bench


def apply_tokens(s: str) -> str:
    for a, b in REPLACEMENTS:
        s = s.replace(a, b)
    return s


def in_scope(path: str) -> bool:
    if path in INCLUDE_FILES:
        return True
    return any(path.startswith(p) for p in INCLUDE_PREFIXES)


def tracked_files():
    # -z: NUL-separated, UNquoted (default core.quotePath would octal-escape
    # non-ASCII names like the Chinese docs/canon/*.md and break open()/git mv).
    out = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO)
    return [b.decode("utf-8", "surrogateescape") for b in out.split(b"\x00") if b]


def scoped_files(extra_prefix=None):
    files = [f for f in tracked_files() if in_scope(f)]
    if extra_prefix:
        files = [f for f in files if f.startswith(extra_prefix) or f == extra_prefix]
    return files


def is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def do_content(files, dry):
    changed = 0
    for rel in files:
        ap = os.path.join(REPO, rel)
        try:
            with open(ap, "rb") as fh:
                data = fh.read()
        except (FileNotFoundError, IsADirectoryError):
            continue
        if is_binary(data):
            continue
        text = data.decode("utf-8", "surrogateescape")
        new = collapse_block_namespace(apply_tokens(text))
        if new != text:
            changed += 1
            if not dry:
                with open(ap, "wb") as fh:
                    fh.write(new.encode("utf-8", "surrogateescape"))
    return changed


def do_moves(files, dry):
    moves = []
    for rel in files:
        newrel = apply_tokens(rel)
        if newrel != rel:
            moves.append((rel, newrel))
    # Longest paths first so nested files move before their dirs are implicitly gone.
    moves.sort(key=lambda m: len(m[0]), reverse=True)
    for old, new in moves:
        if dry:
            continue
        os.makedirs(os.path.dirname(os.path.join(REPO, new)), exist_ok=True)
        subprocess.check_call(["git", "mv", old, new], cwd=REPO)
    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--content", action="store_true")
    ap.add_argument("--moves", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--scope", default=None, help="restrict to a path prefix (A-segment)")
    ap.add_argument("--collapse-ns", action="store_true",
                    help="repair pass for an already-renamed tree that has the doubled "
                         "`weft::weft` namespace: collapse to single `weft` + de-dup block form")
    args = ap.parse_args()

    if args.collapse_ns:
        files = scoped_files(args.scope)
        n = 0
        for rel in files:
            ap_ = os.path.join(REPO, rel)
            try:
                with open(ap_, "rb") as fh:
                    data = fh.read()
            except (FileNotFoundError, IsADirectoryError):
                continue
            if is_binary(data):
                continue
            text = data.decode("utf-8", "surrogateescape")
            new = collapse_block_namespace(text.replace("weft::weft", "weft"))
            if new != text:
                n += 1
                with open(ap_, "wb") as fh:
                    fh.write(new.encode("utf-8", "surrogateescape"))
        print(f"[collapse-ns] repaired {n} files")
        return

    files = scoped_files(args.scope)
    print(f"[scope] {len(files)} tracked files"
          + (f" under {args.scope!r}" if args.scope else " (full functional surface)"))

    if args.plan or not (args.content or args.moves or args.all):
        n_content = do_content(files, dry=True)
        moves = do_moves(files, dry=True)
        print(f"[plan] content would change: {n_content} files")
        print(f"[plan] path moves: {len(moves)}")
        for old, new in sorted(moves):
            print(f"    MOVE  {old}  ->  {new}")
        return

    if args.content or args.all:
        n = do_content(files, dry=False)
        print(f"[content] rewrote {n} files")
    if args.moves or args.all:
        moves = do_moves(files, dry=False)
        print(f"[moves] git mv'd {len(moves)} paths")


if __name__ == "__main__":
    main()
