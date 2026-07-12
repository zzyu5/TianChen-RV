#!/usr/bin/env bash
# [WORK-ITEM-K1-KQUANT-E2E] Isolated build of two libggml-cpu variants on k1 that
# differ ONLY in the q4_K prefill-GEMM dispatch (repack.cpp GEN line ~4620):
#   REPACK = stock as-shipped (case256 -> ggml_gemm/gemv_q4_K_16x1_q8_K repack)  [OFF baseline == /data/k1build-stock]
#   VECDOT = q4_K repack disabled -> generic block-dot ggml_vec_dot_q4_K_q8_K     [forced fallback]
# Both compiled by clang-18 (k1 stock compiler) => COMPILER-SYMMETRIC. This is the
# k1-clang analog of the rvv-gcc M2-q4_K measurement (our-repack(gcc)/stock-vecdot(gcc)=0.42x):
# it swaps gcc->clang on the exact repack-vs-vec_dot contrast to resolve
# gcc-742-spill-death vs intrinsic-weight-reconstruction-overhead.
#
# ISOLATION: never edits Line B's shared source (/home/bianbu/tcrv-k1-llama). A private
# edited COPY of GEN repack.cpp is compiled into a dedicated build dir /data/build-k1-workitem
# (cp -a of /data/k1build-stock). Live stock lib in /data/k1build-stock untouched.
set -e
SRC=/home/bianbu/tcrv-k1-llama/ggml/src/ggml-cpu/repack.cpp
BASE_MD5=3cac40aa55aece1f69e3d08d4e7e9ae2      # shared source baseline (must be preserved)
STOCK=/data/k1build-stock
BUILD=/data/build-k1-workitem
WK=/data/wk
OBJ=$BUILD/ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/repack.cpp.o

test "$(md5sum "$SRC" | awk '{print $1}')" = "$BASE_MD5" || { echo "SRC not at baseline -- ABORT"; exit 10; }
mkdir -p "$WK"
rm -rf "$BUILD"; cp -a "$STOCK" "$BUILD"
cp -a "$BUILD/bin/libggml-cpu.so.0.15.1" "$WK/libggml-cpu.so.REPACK"   # == stock as-shipped

# private edited copy: disable q4_K case256 repack route
cp "$SRC" "$WK/repack_vecdot.cpp"
python3 - <<'PY'
p="/data/wk/repack_vecdot.cpp"; s=open(p).read()
old="                case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; }"
new="                case 256:  { break; } /* TCRV-WORKITEM VECDOT-FORCED: q4_K repack disabled -> generic vec_dot */"
assert s.count(old)==1, f"anchor count={s.count(old)}"
open(p,"w").write(s.replace(old,new))
PY
# recompile ONLY repack.cpp.o (exact stock compile flags), relink ggml-cpu
/usr/bin/clang++-18 -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_SCHED_MAX_COPIES=4 -DGGML_SHARED \
  -DGGML_USE_CPU_REPACK -DGGML_USE_LLAMAFILE -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -Dggml_cpu_EXPORTS \
  -I/home/bianbu/tcrv-k1-llama/ggml/src/.. -I/home/bianbu/tcrv-k1-llama/ggml/src/. \
  -I/home/bianbu/tcrv-k1-llama/ggml/src/ggml-cpu -I/home/bianbu/tcrv-k1-llama/ggml/src/../include \
  -fno-integrated-as -O3 -DNDEBUG -std=gnu++20 -fPIC \
  -march=rv64gcv_zfh_zvfh_zicbop_zihintpause -mabi=lp64d -o "$OBJ" -c "$WK/repack_vecdot.cpp"
( cd "$BUILD/ggml/src" && bash CMakeFiles/ggml-cpu.dir/link.txt )
cp -a "$BUILD/bin/libggml-cpu.so.0.15.1" "$WK/libggml-cpu.so.VECDOT"
# leave build lib at REPACK (pristine)
cp -f "$WK/libggml-cpu.so.REPACK" "$BUILD/bin/libggml-cpu.so.0.15.1"
echo "REPACK md5=$(md5sum "$WK/libggml-cpu.so.REPACK"|awk '{print $1}')"
echo "VECDOT md5=$(md5sum "$WK/libggml-cpu.so.VECDOT"|awk '{print $1}')"
echo "SRC restored-check md5=$(md5sum "$SRC"|awk '{print $1}')  (expect $BASE_MD5)"
