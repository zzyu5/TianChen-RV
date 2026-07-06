#!/usr/bin/env bash
# new-line-worktree.sh — start a parallel work line as an isolated git worktree
# branched from the CURRENT HEAD, with its own out-of-tree build dir.
#
# Roots out the two structural reasons parallel lines kept failing:
#   (②)  A worktree branched from a STALE base (an old origin/main point) lacks
#        the in-flight campaign inputs (experiments/sealed/, uncommitted-but-
#        committed-since work) → the line is forced back into the main tree →
#        isolation gone. We branch from HEAD, so the line carries every tracked
#        campaign input as of now.
#   (③)  A single absolute build/ path makes parallel builds clobber one
#        CMakeCache. Each line gets build/<line>/ INSIDE its own worktree with an
#        independent CMakeCache (via configure-line-build.sh).
#
# It also wires the touch-set gate for the new line (per-worktree pre-commit
# symlink + .touch-set/ACTIVE marker + a starter declaration file).
#
# Usage:
#   tools/bench/new-line-worktree.sh <line-name> [--configure|--no-configure]
#
#   <line-name>       [A-Za-z0-9._-]+  (becomes .worktrees/<name> and branch line/<name>)
#   --configure       (default) also configure build/<name>/ now
#   --no-configure    skip the CMake configure step (create worktree only)
#
# Does NOT build (no ninja) and NEVER touches the main tree's working copy.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tools/bench/_build-common.sh
source "$HERE/_build-common.sh"

DO_CONFIGURE=1
LINE=""
for arg in "$@"; do
  case "$arg" in
    --configure)    DO_CONFIGURE=1 ;;
    --no-configure) DO_CONFIGURE=0 ;;
    -h|--help)
      sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    -*) echo "[new-line] unknown flag: $arg" >&2; exit 2 ;;
    *)
      if [ -n "$LINE" ]; then echo "[new-line] extra arg: $arg" >&2; exit 2; fi
      LINE="$arg" ;;
  esac
done

if [ -z "$LINE" ]; then
  echo "usage: $(basename "$0") <line-name> [--configure|--no-configure]" >&2
  exit 2
fi
if ! [[ "$LINE" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[new-line] line-name must match [A-Za-z0-9._-]+ (got '$LINE')" >&2
  exit 2
fi

ROOT="$(bench_main_root)"
WT="$ROOT/.worktrees/$LINE"
BRANCH="line/$LINE"

if [ -e "$WT" ]; then
  echo "[new-line] worktree path already exists: $WT" >&2
  echo "           remove it first:  git worktree remove $WT" >&2
  exit 1
fi
if git -C "$ROOT" show-ref --verify --quiet "refs/heads/$BRANCH"; then
  echo "[new-line] branch already exists: $BRANCH (pick another line name or delete it)" >&2
  exit 1
fi

echo "[new-line] creating worktree '$LINE' from HEAD ($(git -C "$ROOT" rev-parse --short HEAD))" >&2
# --- (②) branch from HEAD, NOT origin: the line gets the full current campaign ---
# git worktree add prints progress to stdout; send it to stderr so our final
# stdout line is just the worktree path.
git -C "$ROOT" worktree add -b "$BRANCH" "$WT" HEAD 1>&2

# --- touch-set: ACTIVE marker (per-worktree) + starter declaration + gate hint ---
# NOTE: we do NOT auto-install the pre-commit hook. `git rev-parse --git-path
# hooks` resolves to the SHARED main .git/hooks, so symlinking there would (a)
# change main-tree commit behavior and (b) dangle once this worktree is removed.
# The operator installs the gate per-worktree (see tools/hooks/README.md).
echo "$LINE" > "$WT/.touch-set/ACTIVE"   # per-worktree, git-ignored
DECL="$WT/.touch-set/$LINE.txt"
if [ ! -f "$DECL" ]; then
  {
    echo "# line: $LINE — DECLARE the path globs this line may write (one per line)."
    echo "# Keep DISJOINT from every other concurrently-active line. Shared files"
    echo "# (ODS / verifier / emitter) must be serialized, not co-owned. Glob syntax:"
    echo "# see .touch-set/README.md. This file is TRACKED — commit it on the line."
    echo "#"
    echo "# e.g.:"
    echo "# lib/Conversion/RVV/"
    echo "# test/Conversion/RVV/*.mlir"
  } > "$DECL"
  echo "[new-line] wrote starter touch-set declaration: .touch-set/$LINE.txt (EDIT before you commit)" >&2
fi

# --- (③) independent out-of-tree build/<line>/ inside the worktree ---
if [ "$DO_CONFIGURE" = "1" ]; then
  "$HERE/configure-line-build.sh" "$LINE" || {
    echo "[new-line] configure failed (worktree still created; re-run configure-line-build.sh $LINE)" >&2
  }
fi

BUILD_DIR="$WT/build/$LINE"
cat >&2 <<EOF

[new-line] ready. Parallel-line startup protocol:
  worktree : $WT   (branch $BRANCH, from HEAD)
  build    : $BUILD_DIR   (independent CMakeCache)
  touch-set: EDIT .touch-set/$LINE.txt to declare this line's write scope.
             activate the pre-commit gate in THIS worktree (see tools/hooks/README.md):
               cd $WT && git config core.hooksPath tools/hooks
  build    : cd $WT
               $HERE/configure-line-build.sh $LINE      # if you used --no-configure
               ninja -C build/$LINE tcrv-opt tcrv-translate
  verify   : byte-exact vs a pinned base — NEVER 'git stash' in the main tree:
               BASE=\$($HERE/byte-exact-baseline.sh <base-commit> tcrv-opt)
               diff <("\$BASE" IN.mlir --pass) <(build/$LINE/bin/tcrv-opt IN.mlir --pass)
  finish   : commit on $BRANCH (explicit paths only), merge back, then:
               git worktree remove $WT
  See docs/method/parallel-build-and-baseline.md for the full protocol.
EOF

# absolute worktree path on stdout for scripting.
echo "$WT"
