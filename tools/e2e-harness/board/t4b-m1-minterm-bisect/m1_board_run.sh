#!/usr/bin/env bash
# [G3-minterm-fix M1] Runs ON the board (rvv). Deploys the INSTRUMENTED S6 q4_K repack-GEMM
# kernel (base md5 90d454da + pure-observation min-term captures) through the SAME reversible
# dispatch patch, rebuilds ggml-cpu, runs the instrumented bisect driver (real ggml_mul_mat,
# min-term exercised, dumps kernel-vs-oracle per (block,row,col)), then objdumps the min-fold.
# Forced A-tree restore via EXIT trap. NO git stash/rm/mv/add/commit. NO emitter source change.
set -uo pipefail
ATREE=/home/ubuntu/tcrv-llamacpp
SCR=/tmp/m1_minterm_board
BUILD=$ATREE/build-gcc15-rv64gcv
BIN=$BUILD/bin
INCG=$ATREE/ggml/include
GEN=$ATREE/ggml/src/ggml-cpu/repack.cpp
ARCH=$ATREE/ggml/src/ggml-cpu/arch/riscv/repack.cpp
CORE=${CORE:-8}
BASE_GEN=deb61a29dd079440ffdc8996b5bd2fa1
BASE_ARCH=99131cf791e30348b588423b2388e0b8
KSYM=tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K

source /opt/tcrv-toolchains/env.sh
export LIBRARY_PATH=/opt/tcrv-toolchains/gcc-15.2.0/lib:${LIBRARY_PATH:-}

restore() {
  echo ""; echo "=== [restore] forced A-tree restore ==="
  if [ -f "$SCR/repack.cpp.ORIG" ] && [ -f "$SCR/arch_riscv_repack.cpp.ORIG" ]; then
    cp "$SCR/repack.cpp.ORIG" "$GEN"; cp "$SCR/arch_riscv_repack.cpp.ORIG" "$ARCH"
  else echo "  (no backups -- nothing to restore)"; return; fi
  echo "  restored md5:"; md5sum "$GEN" "$ARCH"
  echo "  expect GEN  $BASE_GEN"; echo "  expect ARCH $BASE_ARCH"
  echo "=== [restore] rebuild ggml-cpu pristine ==="
  if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/m1_restore_build.log 2>&1; then echo "  restore build OK"
  else echo "  restore build FAIL"; tail -25 /tmp/m1_restore_build.log; fi
  echo "  golden symbol in pristine .so (expect 0):"; nm -C "$BIN/libggml-cpu.so.0.15.1" 2>/dev/null | grep -c "$KSYM" || true
  echo "  banner in pristine .so (expect 0):"; strings "$BIN/libggml-cpu.so.0.15.1" 2>/dev/null | grep -c "TCRV EMITTED GEMM(q4_K_16x1 VLEN128" || true
}
trap restore EXIT

echo "[m1] board=rvv VLEN128 gcc-15.2.0 ATREE=$ATREE core=$CORE loadavg=$(cat /proc/loadavg)"
echo "=== [0] baseline verify + backup 2 patch-target files ==="
G0=$(md5sum "$GEN"|awk '{print $1}'); A0=$(md5sum "$ARCH"|awk '{print $1}')
echo "  GEN  now=$G0  base=$BASE_GEN"; echo "  ARCH now=$A0  base=$BASE_ARCH"
if [ "$G0" != "$BASE_GEN" ] || [ "$A0" != "$BASE_ARCH" ]; then
  echo "  *** A-tree NOT at baseline -- ABORT (no patch/backup) ***"; exit 10; fi
cp "$GEN" "$SCR/repack.cpp.ORIG"; cp "$ARCH" "$SCR/arch_riscv_repack.cpp.ORIG"
echo "  backups captured."
echo "=== [0b] instrumented kernel fingerprint (base md5 must be 90d454da when captures stripped) ==="
md5sum "$SCR/instr_q4K.inc"
grep -c "$KSYM" "$SCR/instr_q4K.inc"
grep -c "G3 M1] tile0" "$SCR/instr_q4K.inc"

echo "=== [1] apply M1 patch (instrumented kernel) ==="
python3 "$SCR/m1_patch.py" || { echo "PATCH FAILED"; exit 11; }

echo "=== [2] rebuild ggml-cpu ==="
if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/m1_patch_build.log 2>&1; then echo "  patch build OK"
else echo "  patch build FAIL"; tail -40 /tmp/m1_patch_build.log; exit 20; fi
ls -la "$BIN/libggml-cpu.so.0.15.1"
echo "  golden symbol (expect >=1):"; nm -C "$BIN/libggml-cpu.so.0.15.1" | grep -c "$KSYM"
echo "  banner (expect >=1):"; strings "$BIN/libggml-cpu.so.0.15.1" | grep -c "TCRV EMITTED GEMM(q4_K_16x1 VLEN128"
echo "  capture globals exported (expect tcrv_min_i tcrv_dmin present):"; nm -D "$BIN/libggml-cpu.so.0.15.1" | grep -E "tcrv_min_i|tcrv_dmin|tcrv_main_i|tcrv_cap_want" || echo "  (WARN none exported dynamically)"

echo "=== [3] compile m1_probe driver ==="
if g++ -O2 -march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d \
     -I"$INCG" -I"$SCR" "$SCR/m1_probe.cpp" -o "$SCR/m1_probe" \
     -L"$BIN" -Wl,-rpath,"$BIN" -lggml-cpu -lggml-base -lggml -lm 2>/tmp/m1_drv_build.log; then echo "  driver compile OK"
else echo "  driver compile FAIL"; cat /tmp/m1_drv_build.log; exit 30; fi

echo "=== [4] RUN m1_probe (real ggml_mul_mat -> instrumented kernel, min-term bisect) ==="
LD_LIBRARY_PATH="$BIN" taskset -c "$CORE" "$SCR/m1_probe"
echo "RUN_EXIT=$?"

echo ""
echo "=== [5] objdump min-fold region (instruction-level evidence) ==="
# Slice the kernel function; show the min-fold signature: e32 vl=8 vwmacc.vx (min acc) + vfnmsac (float fold)
objdump -d "$BIN/libggml-cpu.so.0.15.1" > /tmp/m1_kernel.objdump 2>/dev/null
echo "  --- vsetivli/vsetvli setting vl for the 8-lane e32m2 strips (grep zero=..e32,m2) ---"
grep -nE "vsetivli.*e32,m2|vsetvli.*e32,m2" /tmp/m1_kernel.objdump | head -6
echo "  --- vwmacc.vx (integer min accumulator, per-strip) count in .so ---"
grep -cE "vwmacc\.vx" /tmp/m1_kernel.objdump
echo "  --- vfnmsac.vv (float dmin*min fold) occurrences ---"
grep -nE "vfnmsac\.vv" /tmp/m1_kernel.objdump | head -12
echo "  --- context around first vfnmsac.vv (the min fold) ---"
FIRST=$(grep -nE "vfnmsac\.vv" /tmp/m1_kernel.objdump | head -1 | cut -d: -f1)
if [ -n "$FIRST" ]; then sed -n "$((FIRST-14)),$((FIRST+2))p" /tmp/m1_kernel.objdump; fi
echo "objdump saved to /tmp/m1_kernel.objdump ($(wc -l </tmp/m1_kernel.objdump) lines)"

echo "=== [6] done -- EXIT trap force-restores A-tree + rebuilds pristine ==="
