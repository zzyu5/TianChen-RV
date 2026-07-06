#!/usr/bin/env bash
# run_silicon_batch2.sh -- silicon-validation batch-2: close the iq3_xxs + iq2_xxs
# flip debts (2 -> 0). Same discipline as batch-1.
#
# Pipeline:
#   1. (host) export each front-door-CONSTRUCTED kernel to C:
#        tcrv-opt <fixture.mlir> <materialize-front-door> --tcrv-rvv-lower-to-emitc \
#          | mlir-translate --mlir-to-cpp  > kernels/<fmt>.kernel.c
#   2. (board) compile the pinned no-FMA oracle from ggml-cpu/quants.c at -ffp-contract=off,
#      compile the exported kernels (C++, extern "C") + the verifier driver,
#      link with --gc-sections, run bit-exact vs oracle over multiple seeds.
#
# The oracle = ggml's OWN generic scalar ggml_vec_dot_<fmt>_generic, recompiled
# -ffp-contract=off so the sumf reduction is scalar / left-assoc / non-FMA.
set -u
HOST="${HOST:-rvv}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # this harness dir (driver lives here)
REPO="$(cd "$HERE/../../.." && pwd)"                        # repo root
CELL="$REPO/experiments/silicon-validation-batch-2"        # DATA cell: kernels/ + results/ stay under experiments/
GGML="${GGML:-/home/ubuntu/tcrv-llamacpp}"
OPT="$REPO/build/bin/tcrv-opt"
MT="${MT:-/usr/lib/llvm-20/bin/mlir-translate}"
RDIR="/tmp/sv_batch2_build"
MARCH="rv64gcv_zfh_zvfh_zicbop_zihintpause"
FF="-O2 -march=$MARCH -mabi=lp64d -ffp-contract=off"

# ---- step 1 (host): export kernels to C ----
export_c () { # <out.c> <mlir> <passes...>
  local out="$1" mlir="$2"; shift 2
  "$OPT" "$REPO/$mlir" "$@" --tcrv-rvv-lower-to-emitc | "$MT" --mlir-to-cpp > "$out"
}
mkdir -p "$CELL/kernels"
export_c "$CELL/kernels/iq3_xxs.kernel.c" test/Target/RVV/iq3-xxs-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir --tcrv-rvv-materialize-iq3-xxs-q8-k-block-dot-source-front-door
export_c "$CELL/kernels/iq2_xxs.kernel.c" test/Target/RVV/iq2-xxs-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir --tcrv-rvv-materialize-iq2-xxs-q8-k-block-dot-source-front-door
echo "exported: $(wc -l < "$CELL/kernels/iq3_xxs.kernel.c") + $(wc -l < "$CELL/kernels/iq2_xxs.kernel.c") lines"

# ---- step 2 (board): build + run ----
ssh "$HOST" "mkdir -p $RDIR"
scp -q "$HERE/bd_verify_driver_iq23.c" \
       "$CELL"/kernels/{iq3_xxs,iq2_xxs}.kernel.c "$HOST:$RDIR/"
ssh "$HOST" "bash -s" <<REMOTE
set -e; cd $RDIR
CLANG=\$(command -v clang-17 || command -v clang); CXX=\$(command -v clang++-17 || command -v clang++)
echo "board clang: \$(\$CLANG --version | head -1)"
FF="$FF"
\$CLANG \$FF -ffunction-sections -fdata-sections -fno-stack-protector \
  -I$GGML/ggml/src/.. -I$GGML/ggml/src/. -I$GGML/ggml/src/ggml-cpu -I$GGML/ggml/src/../include \
  -c $GGML/ggml/src/ggml-cpu/quants.c -o quants_oracle.o
for k in iq3_xxs iq2_xxs; do \$CXX \$FF -c \$k.kernel.c -o \$k.kernel.o; done
\$CLANG \$FF -c bd_verify_driver_iq23.c -o bd23.o
\$CXX \$FF -Wl,--gc-sections bd23.o iq3_xxs.kernel.o iq2_xxs.kernel.o quants_oracle.o -lm -o bd23_verify
echo "== RUN iq3_xxs =="; for s in 0x1234567 0xABCDEF01 0x55AA55AA; do ./bd23_verify iq3_xxs 4096 256 \$s; done
echo "== RUN iq2_xxs =="; for s in 0x1234567 0xABCDEF01 0x55AA55AA; do ./bd23_verify iq2_xxs 4096 256 \$s; done
REMOTE
