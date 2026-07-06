#!/usr/bin/env bash
# byte-exact-baseline.sh — produce (and cache) a pinned-base build product for
# byte-exact verification, WITHOUT ever touching the main tree or `git stash`.
#
# ── Why this exists (root cause ①) ──────────────────────────────────────────
# The old byte-exact gate built the "BEFORE" side by `git stash`-ing the working
# edits, rebuilding at HEAD, then un-stashing. `git stash` is a GLOBAL tree
# operation: it swept away *other* parallel lines' uncommitted work (empirically
# clobbered a sibling line's settings.json / .gitignore). So a byte-exact run in
# the main tree made parallel lines unsafe even with disjoint file sets.
#
# This replaces the stash entirely:
#   * `git worktree add --detach .worktrees/baseline-<sha> <base-commit>` checks
#     the pinned base out into a SEPARATE directory. The main working copy — and
#     every sibling line's uncommitted work — is never touched.
#   * The base product (tcrv-opt / tcrv-translate) is cached by commit hash under
#     .worktrees/cache/<sha>/. A second call for the same <sha> is a cache hit:
#     no rebuild, idempotent.
#
# ── Usage ───────────────────────────────────────────────────────────────────
#   tools/bench/byte-exact-baseline.sh <base-commit> <target> [--dry-run] [--keep-worktree]
#
#   <base-commit>     any git ref (HEAD, a tag, a sha) — pinned by resolved sha
#   <target>          tcrv-opt | tcrv-translate  (ninja target = binary name)
#   --dry-run         print the plan (resolved sha, cache path, build steps) and exit
#   --keep-worktree   leave .worktrees/baseline-<sha> in place (for debugging)
#
# Prints ONLY the absolute path of the cached base binary on stdout, e.g.
#   BASE=$(tools/bench/byte-exact-baseline.sh HEAD tcrv-opt)
# then verify (no stash, no main-tree mutation):
#   diff <("$BASE" IN.mlir --pass) <(build/<line>/bin/tcrv-opt IN.mlir --pass)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tools/bench/_build-common.sh
source "$HERE/_build-common.sh"

DRY_RUN=0
KEEP_WT=0
POS=()
for arg in "$@"; do
  case "$arg" in
    --dry-run)       DRY_RUN=1 ;;
    --keep-worktree) KEEP_WT=1 ;;
    -h|--help)
      sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "[baseline] unknown flag: $arg" >&2; exit 2 ;;
    *)  POS+=("$arg") ;;
  esac
done

if [ "${#POS[@]}" -ne 2 ]; then
  echo "usage: $(basename "$0") <base-commit> <target> [--dry-run] [--keep-worktree]" >&2
  exit 2
fi
BASE_REF="${POS[0]}"
TARGET="${POS[1]}"
case "$TARGET" in
  tcrv-opt|tcrv-translate) : ;;
  *) echo "[baseline] target must be tcrv-opt or tcrv-translate (got '$TARGET')" >&2; exit 2 ;;
esac

ROOT="$(bench_main_root)"

# Pin the base by its resolved full sha, so the cache key is unambiguous.
SHA="$(git -C "$ROOT" rev-parse --verify "${BASE_REF}^{commit}" 2>/dev/null)" || {
  echo "[baseline] cannot resolve base-commit '$BASE_REF' to a commit" >&2; exit 1; }

CACHE_DIR="$ROOT/.worktrees/cache/$SHA"
PRODUCT="$CACHE_DIR/$TARGET"
BASE_WT="$ROOT/.worktrees/baseline-$SHA"
BUILD_DIR="$BASE_WT/build"

if [ "$DRY_RUN" = "1" ]; then
  {
    echo "[baseline] DRY RUN"
    echo "  base ref     : $BASE_REF"
    echo "  resolved sha : $SHA"
    echo "  target       : $TARGET"
    echo "  cache product: $PRODUCT"
    if [ -f "$PRODUCT" ]; then
      echo "  status       : CACHE HIT (would return immediately, no build)"
    else
      echo "  status       : CACHE MISS — would:"
      echo "    git worktree add --detach $BASE_WT $SHA"
      echo "    cmake -S $BASE_WT -B $BUILD_DIR -G Ninja <llvm/mlir/launcher flags>"
      echo "    ninja -C $BUILD_DIR $TARGET"
      echo "    cp $BUILD_DIR/bin/$TARGET $PRODUCT"
      echo "    git worktree remove --force $BASE_WT"
    fi
    echo "  main tree    : NOT touched (no git stash, ever)"
  } >&2
  # Even in dry-run, print the (future) product path so callers can compose.
  echo "$PRODUCT"
  exit 0
fi

# ── Cache hit: idempotent, no rebuild ───────────────────────────────────────
if [ -f "$PRODUCT" ]; then
  echo "[baseline] cache HIT for $SHA/$TARGET — no rebuild." >&2
  echo "$PRODUCT"
  exit 0
fi

# ── Cache miss: build the base in a DETACHED worktree, then cache the binary ──
echo "[baseline] cache MISS for $SHA/$TARGET — building pinned base (main tree untouched)." >&2

cleanup_wt() {
  if [ "$KEEP_WT" = "0" ] && [ -d "$BASE_WT" ]; then
    git -C "$ROOT" worktree remove --force "$BASE_WT" 2>/dev/null || true
  fi
}
trap cleanup_wt EXIT

{
  if [ ! -d "$BASE_WT" ]; then
    git -C "$ROOT" worktree add --detach "$BASE_WT" "$SHA"
  else
    echo "[baseline] reusing existing detached worktree $BASE_WT" >&2
  fi
  bench_configure "$BASE_WT" "$BUILD_DIR"
  ninja -C "$BUILD_DIR" "$TARGET"
} 1>&2   # keep stdout clean — only the cached path goes there

BUILT="$BUILD_DIR/bin/$TARGET"
if [ ! -x "$BUILT" ]; then
  echo "[baseline] expected built binary not found: $BUILT" >&2
  exit 1
fi

mkdir -p "$CACHE_DIR"
cp -f "$BUILT" "$PRODUCT"
echo "[baseline] cached base $TARGET -> $PRODUCT" >&2

echo "$PRODUCT"
