#!/usr/bin/env bash
# board_run.sh <rvv|k1> [kernel_src] -- B线 block2 P3: DEPLOYED-EMIT tq1_0 vec_dot direct
# M=1 board measure (P2 iq2 paradigm). scp the deployed-emit leaf (.cpp) + driver + remote
# script, build compiler-symmetric (clang-18 both sides) and run byte-exact verify + cold.
# Repo-side零板端产物: only pulls the seal log back into raw/.
set -uo pipefail
BOARD="${1:?usage: board_run.sh <rvv|k1> [kernel_src]}"
KERNEL_SRC="${2:-kernels/tq1_0_deployed_emit.cpp}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDIR=/tmp/b_block2_p3_tq10_deployed

case "$BOARD" in
  rvv)  # deployed clang-18.1.8 symmetric lib as opponent; env.sh gives clang-18 + gcc runtime
    ENVSH=/opt/tcrv-toolchains/env.sh
    CC=clang
    FACTORY_MODE=lib
    GGML=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin
    MARCH=rv64gcv_zfh_zvfh_zba_zbb
    CORES="8 9 10 11 12 13 14 15"
    REMOTE_ENV="ENVSH='$ENVSH' CC='$CC' MARCH='$MARCH' RDIR='$RDIR' CORES='$CORES' FACTORY_MODE='$FACTORY_MODE' GGML='$GGML' KERNEL_SRC='$KERNEL_SRC'" ;;
  k1)   # clang-18 compiles ours + factory from ggml source (compiler-symmetric)
    CC=/usr/bin/clang-18
    R=/home/bianbu/tcrv-k1-llama
    FACTORY_MODE=source
    MARCH=rv64gcv_zfh_zvfh_zicbop_zihintpause
    CORES="0 1 2 3 4 5 6 7"
    REMOTE_ENV="CC='$CC' R='$R' MARCH='$MARCH' RDIR='$RDIR' CORES='$CORES' FACTORY_MODE='$FACTORY_MODE' KERNEL_SRC='$KERNEL_SRC'" ;;
  *) echo "unknown board $BOARD"; exit 2 ;;
esac

echo "[$BOARD] scp harness -> $BOARD:$RDIR"
ssh "$BOARD" "mkdir -p $RDIR/kernels"
scp -q "$HERE/tq10_vecdot_driver.c" "$BOARD:$RDIR/"
scp -q "$HERE/remote_build_run.sh" "$BOARD:$RDIR/"
scp -q "$HERE"/kernels/tq1_0_deployed_emit.cpp "$BOARD:$RDIR/kernels/"

echo "[$BOARD] build + run (compiler-symmetric clang-18) kernel=$KERNEL_SRC"
ssh "$BOARD" "$REMOTE_ENV bash $RDIR/remote_build_run.sh"

TAG=$(basename "$KERNEL_SRC" | sed 's/\.[^.]*$//')
echo "[$BOARD] pull seal"
mkdir -p "$HERE/raw"
scp -q "$BOARD:$RDIR/seal.txt" "$HERE/raw/${BOARD}_${TAG}_seal.txt" 2>/dev/null || echo "  (seal pull skipped)"
echo "[$BOARD] done -> $HERE/raw/${BOARD}_${TAG}_seal.txt"
