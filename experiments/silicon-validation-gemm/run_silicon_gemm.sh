#!/usr/bin/env bash
# run_silicon_gemm.sh -- reproduce the constructed q4_0 REPACK GEMM (PREFILL)
# silicon validation on ssh rvv. Sibling of ../silicon-validation-batch-1.
#
# Pipeline:
#   1. (host) export the front-door-CONSTRUCTED GEMM kernel to C:
#        tcrv-opt <gemm-full-pipeline-export mlir> \
#          --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc \
#          | mlir-translate --mlir-to-cpp  > kernels/q4_0_repack_gemm.kernel.c
#   2. (board) compile the pinned no-FMA oracle from ggml-cpu/quants.c at
#      -ffp-contract=off (scalar / left-assoc / non-FMA ggml_vec_dot_q4_0_q8_0_generic),
#      compile the exported GEMM kernel (C++, extern "C") + the driver, link with
#      --gc-sections, run bit-exact vs oracle over several (K,nr,nc,seed) shapes.
set -u
HOST="${HOST:-rvv}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CELL="$REPO/experiments/silicon-validation-gemm"
GGML="${GGML:-/home/ubuntu/tcrv-llamacpp}"
OPT="$REPO/build/bin/tcrv-opt"
MT="${MT:-/usr/lib/llvm-20/bin/mlir-translate}"
MLIR="test/Target/RVV/q4-0-q8-0-repack-gemm-full-pipeline-export-e2e.mlir"
RDIR="/tmp/sv_gemm_build"
MARCH="rv64gcv_zfh_zvfh_zicbop_zihintpause"
FF="-O2 -march=$MARCH -mabi=lp64d -ffp-contract=off"

# ---- step 1 (host): export constructed GEMM kernel to C ----
mkdir -p "$CELL/kernels"
"$OPT" "$REPO/$MLIR" --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc \
  | "$MT" --mlir-to-cpp > "$CELL/kernels/q4_0_repack_gemm.kernel.c"

# ---- step 2 (board): build + run ----
ssh "$HOST" "mkdir -p $RDIR"
scp -q "$CELL/gemm_verify_driver.c" "$CELL/kernels/q4_0_repack_gemm.kernel.c" "$HOST:$RDIR/"
ssh "$HOST" "bash -s" <<REMOTE
set -e; cd $RDIR
CLANG=\$(command -v clang-17); CXX=\$(command -v clang++-17 || command -v clang++)
FF="$FF"
\$CLANG \$FF -ffunction-sections -fdata-sections -fno-stack-protector \
  -I$GGML/ggml/src/.. -I$GGML/ggml/src/. -I$GGML/ggml/src/ggml-cpu -I$GGML/ggml/src/../include \
  -c $GGML/ggml/src/ggml-cpu/quants.c -o quants_oracle.o
\$CXX \$FF -c q4_0_repack_gemm.kernel.c -o gemm.kernel.o
\$CLANG \$FF -c gemm_verify_driver.c -o gvd.o
\$CXX \$FF -Wl,--gc-sections gvd.o gemm.kernel.o quants_oracle.o -lm -o gemm_verify
# shapes: K, nr(mult4), nc(mult16), trials, seed
./gemm_verify 4096 4  128 2 0xBEEF01
./gemm_verify 2048 8  64  2 0xC0FFEE
./gemm_verify 4096 16 256 1 0x99887766
./gemm_verify 128  4  16  4 0x1234567
REMOTE
