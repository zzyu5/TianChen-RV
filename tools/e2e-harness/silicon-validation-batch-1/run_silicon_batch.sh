#!/usr/bin/env bash
# run_silicon_batch.sh -- reproduce silicon-validation-batch-1 on ssh rvv.
#
# Pipeline:
#   1. (host) export each front-door-CONSTRUCTED kernel to C:
#        tcrv-opt <mlir> <front-door/lower-quant-contraction> --tcrv-rvv-lower-to-emitc \
#          | mlir-translate --mlir-to-cpp  > kernels/<fmt>.kernel.c
#   2. (board) compile the pinned no-FMA oracle from ggml-cpu/quants.c at -ffp-contract=off,
#      compile the exported kernels (C++, extern "C") + the verifier drivers,
#      link with --gc-sections, run bit-exact vs oracle.
#
# The oracle = ggml's OWN generic scalar ggml_vec_dot_<fmt>_generic, recompiled
# -ffp-contract=off so the sumf reduction is scalar / left-assoc / non-FMA.
set -u
HOST="${HOST:-rvv}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # this harness dir (drivers live here, tools/)
REPO="$(cd "$HERE/../../.." && pwd)"                        # repo root (tools/e2e-harness/<cell> -> ../../..)
CELL="$REPO/experiments/silicon-validation-batch-1"        # DATA cell: kernels/ + results/ stay under experiments/
GGML="${GGML:-/home/ubuntu/tcrv-llamacpp}"
OPT="$REPO/build/bin/tcrv-opt"
MT="${MT:-/usr/lib/llvm-20/bin/mlir-translate}"
RDIR="/tmp/sv_batch1_build"
MARCH="rv64gcv_zfh_zvfh_zicbop_zihintpause"
FF="-O2 -march=$MARCH -mabi=lp64d -ffp-contract=off"

# ---- step 1 (host): export kernels to C ----
export_c () { # <out.c> <mlir> <passes...>
  local out="$1" mlir="$2"; shift 2
  "$OPT" "$REPO/$mlir" "$@" --tcrv-rvv-lower-to-emitc | "$MT" --mlir-to-cpp > "$out"
}
mkdir -p "$CELL/kernels"
export_c "$CELL/kernels/iq4_nl.kernel.c" test/Target/RVV/iq4-nl-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir      --tcrv-rvv-materialize-iq4-nl-q8-0-block-dot-source-front-door
export_c "$CELL/kernels/iq1_s.kernel.c"  test/Target/RVV/iq1-s-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir --tcrv-rvv-materialize-iq1-s-q8-k-block-dot-source-front-door
export_c "$CELL/kernels/iq1_m.kernel.c"  test/Target/RVV/iq1-m-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir --tcrv-rvv-materialize-iq1-m-q8-k-block-dot-source-front-door
export_c "$CELL/kernels/q4_0_repack.kernel.c" test/Target/RVV/q4-0-q8-0-repack-gemv-full-pipeline-export-e2e.mlir       --tcrv-rvv-lower-quant-contraction=march=rv64gcv

# ---- step 2 (board): build + run ----
ssh "$HOST" "mkdir -p $RDIR"
scp -q "$HERE/bd_verify_driver.c" "$HERE/q4_0_repack_verify_driver.c" \
       "$CELL"/kernels/{iq4_nl,iq1_s,iq1_m,q4_0_repack}.kernel.c "$HOST:$RDIR/"
ssh "$HOST" "bash -s" <<REMOTE
set -e; cd $RDIR
CLANG=\$(command -v clang-17); CXX=\$(command -v clang++-17 || command -v clang++)
FF="$FF"
\$CLANG \$FF -ffunction-sections -fdata-sections -fno-stack-protector \
  -I$GGML/ggml/src/.. -I$GGML/ggml/src/. -I$GGML/ggml/src/ggml-cpu -I$GGML/ggml/src/../include \
  -c $GGML/ggml/src/ggml-cpu/quants.c -o quants_oracle.o
for k in iq4_nl iq1_s iq1_m q4_0_repack; do \$CXX \$FF -c \$k.kernel.c -o \$k.kernel.o; done
\$CLANG \$FF -c bd_verify_driver.c -o bd.o
\$CLANG \$FF -c q4_0_repack_verify_driver.c -o rd.o
\$CXX \$FF -Wl,--gc-sections bd.o iq4_nl.kernel.o iq1_s.kernel.o iq1_m.kernel.o quants_oracle.o -lm -o bd_verify
\$CXX \$FF -Wl,--gc-sections rd.o q4_0_repack.kernel.o quants_oracle.o -lm -o rgemv
for f in iq4_nl iq1_s iq1_m; do ./bd_verify \$f 4096 256 0x1234567; ./bd_verify \$f 4096 256 0xABCDEF01; ./bd_verify \$f 8192 256 0x55AA55AA; done
./rgemv 4096 32 64 0xBEEF01; ./rgemv 2048 16 128 0xC0FFEE; ./rgemv 4096 16 128 0x99887766
REMOTE
