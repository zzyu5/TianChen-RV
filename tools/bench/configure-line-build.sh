#!/usr/bin/env bash
# configure-line-build.sh — configure an INDEPENDENT out-of-tree build for one
# work line's worktree.
#
# This is the §3.2 mechanism (each line an independent CMakeCache) as its own
# reusable step, so a line can be re-configured without recreating the worktree.
# The build dir lives INSIDE the worktree at build/<line>/, giving it a distinct
# absolute path and its own CMakeCache — parallel `ninja` runs never clobber each
# other (root cause ③).
#
# Configure only — this does NOT run ninja (leave the compile to the caller).
#
# Usage:
#   tools/bench/configure-line-build.sh <line-name> [build-dir]
#
#   <line-name>   the line whose worktree (.worktrees/<line>) to configure
#   [build-dir]   override the build dir (default: <worktree>/build/<line>)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tools/bench/_build-common.sh
source "$HERE/_build-common.sh"

LINE="${1:-}"
if [ -z "$LINE" ]; then
  echo "usage: $(basename "$0") <line-name> [build-dir]" >&2
  exit 2
fi

ROOT="$(bench_main_root)"
WT="$ROOT/.worktrees/$LINE"
if [ ! -d "$WT" ]; then
  echo "[configure-line] no worktree for line '$LINE' at $WT" >&2
  echo "                 create it first:  $HERE/new-line-worktree.sh $LINE" >&2
  exit 1
fi

BUILD_DIR="${2:-$WT/build/$LINE}"
bench_configure "$WT" "$BUILD_DIR"

cat >&2 <<EOF
[configure-line] configured. Build with:
    ninja -C $BUILD_DIR tcrv-opt tcrv-translate
    ninja -C $BUILD_DIR check-tianchenrv
EOF

echo "$BUILD_DIR"   # build dir on stdout for scripting
