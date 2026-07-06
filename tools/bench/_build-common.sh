#!/usr/bin/env bash
# _build-common.sh — shared helpers for the parallel-build / byte-exact infra.
#
# Sourced (not executed) by:
#   * new-line-worktree.sh / configure-line-build.sh — per-line out-of-tree build
#   * byte-exact-baseline.sh                          — pinned-base cached build
#
# It exists so the CMake configure incantation (LLVM/MLIR dirs, generator,
# compiler-cache launcher) is *identical* between a work line's build and the
# byte-exact baseline build. If they diverged, a byte-diff of two emits could
# reflect a toolchain difference rather than a source change — silently voiding
# the gate. Keep all "which toolchain / how to configure" logic here, once.
#
# Convention: value-returning helpers print ONLY the value to stdout; all human
# notes go to stderr. Callers that need a clean stdout (byte-exact-baseline)
# redirect the noisy steps themselves.

set -euo pipefail

# Absolute path to the MAIN worktree root, regardless of which (main or linked)
# worktree we are invoked from. All shared state (.worktrees/, cache) is anchored
# here so parallel lines see one cache.
bench_main_root() {
  local cdir
  cdir="$(cd "$(git rev-parse --git-common-dir)" 2>/dev/null && pwd)" || {
    echo "[bench] not inside a git repo" >&2; return 1; }
  dirname "$cdir"   # <main-root>/.git -> <main-root>
}

# Echo the -DLLVM_DIR / -DMLIR_DIR flags. Prefer the values baked into the
# existing main-tree build/CMakeCache.txt (so a line build matches the toolchain
# the repo is actually using); fall back to the README defaults.
bench_llvm_dir_flags() {
  local root cache llvm mlir
  root="$(bench_main_root)"
  cache="$root/build/CMakeCache.txt"
  llvm=""; mlir=""
  if [ -f "$cache" ]; then
    llvm="$(sed -nE 's/^LLVM_DIR(:[^=]*)?=(.+)$/\2/p' "$cache" | head -n1)"
    mlir="$(sed -nE 's/^MLIR_DIR(:[^=]*)?=(.+)$/\2/p' "$cache" | head -n1)"
  fi
  : "${llvm:=/usr/lib/llvm-20/lib/cmake/llvm}"
  : "${mlir:=/usr/lib/llvm-20/lib/cmake/mlir}"
  printf -- '-DLLVM_DIR=%s -DMLIR_DIR=%s' "$llvm" "$mlir"
}

# Echo compiler-cache launcher flags if ccache/sccache is on PATH; otherwise echo
# nothing (and note to stderr). A shared cache lets N line builds + the baseline
# build reuse each other's object files, amortizing the cost of many trees.
bench_launcher_flags() {
  local tool
  for tool in ccache sccache; do
    if command -v "$tool" >/dev/null 2>&1; then
      echo "[bench] using compiler cache: $tool (shared across worktrees)" >&2
      printf -- '-DCMAKE_C_COMPILER_LAUNCHER=%s -DCMAKE_CXX_COMPILER_LAUNCHER=%s' "$tool" "$tool"
      return 0
    fi
  done
  echo "[bench] no ccache/sccache on PATH — builds will not share objects (install one to amortize multi-tree builds)." >&2
  printf ''
}

# bench_configure <src_dir> <build_dir>
# Idempotent CMake configure (re-running is safe; CMake reuses the cache).
bench_configure() {
  local src="$1" build="$2"
  local llvm_flags launcher_flags
  llvm_flags="$(bench_llvm_dir_flags)"
  launcher_flags="$(bench_launcher_flags)"
  echo "[bench] configure  src=$src  build=$build" >&2
  # shellcheck disable=SC2086
  cmake -S "$src" -B "$build" -G Ninja $llvm_flags $launcher_flags
}
