#!/usr/bin/env bash
# [G3-cert-hardening M4] Runs ON the board (rvv, VLEN128). DECISIVE positional:
#   re-emitted q4_K (base md5 90d454da, instrumented) + q5_K (md5 c209226b) kernels are compiled
#   into libggml-cpu.so via a MINIMAL NON-DESTRUCTIVE reversible ARCH-only patch (append include +
#   visibility-default wrappers; ggml_gemm_q4_K_16x1_q8_K + _generic UNTOUCHED). Driver then feeds a
#   REAL Q4_K_M model tensor (dmin!=0) + the SAME ggml mat-quant q8 activation to BOTH our kernel and
#   ggml's own generic, integer bit-exact captures the min-term, and compares. Forced A-tree restore
#   via EXIT trap. NO git stash/rm/mv/add/commit. NO emitter source change.
set -uo pipefail
ATREE=/home/ubuntu/tcrv-llamacpp
SCR=/tmp/m4_decisive
BUILD=$ATREE/build-gcc15-rv64gcv
BIN=$BUILD/bin
INCG=$ATREE/ggml/include
GEN=$ATREE/ggml/src/ggml-cpu/repack.cpp
ARCH=$ATREE/ggml/src/ggml-cpu/arch/riscv/repack.cpp
CORE=${CORE:-8}
MODEL=/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
BASE_GEN=deb61a29dd079440ffdc8996b5bd2fa1
BASE_ARCH=99131cf791e30348b588423b2388e0b8
KSYM4=tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K
KSYM5=tcrv_emitc_ggml_repack_gemm_q5_K_q8_K_kernel_ggml_repack_gemm_q5_K_q8_K

source /opt/tcrv-toolchains/env.sh
export LIBRARY_PATH=/opt/tcrv-toolchains/gcc-15.2.0/lib:${LIBRARY_PATH:-}

restore() {
  echo ""; echo "=== [restore] forced A-tree restore (ARCH only) ==="
  if [ -f "$SCR/arch_riscv_repack.cpp.ORIG" ]; then cp "$SCR/arch_riscv_repack.cpp.ORIG" "$ARCH"
  else echo "  (no backup -- nothing to restore)"; return; fi
  echo "  restored md5:"; md5sum "$ARCH"; echo "  expect ARCH $BASE_ARCH"
  echo "  GEN untouched md5:"; md5sum "$GEN"; echo "  expect GEN  $BASE_GEN"
  echo "=== [restore] rebuild ggml-cpu pristine ==="
  if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/m4_restore_build.log 2>&1; then echo "  restore build OK"
  else echo "  restore build FAIL"; tail -25 /tmp/m4_restore_build.log; fi
  echo "  our kernels in pristine .so (expect 0/0):"; nm -C "$BIN/libggml-cpu.so.0.15.1" 2>/dev/null | grep -c "$KSYM4" || true; nm -C "$BIN/libggml-cpu.so.0.15.1" 2>/dev/null | grep -c "$KSYM5" || true
  echo "  wrappers in pristine .so (expect 0):"; nm -D "$BIN/libggml-cpu.so.0.15.1" 2>/dev/null | grep -c "tcrv_m4_call" || true
  echo "  pristine .so size:"; stat -c%s "$BIN/libggml-cpu.so.0.15.1"
}
trap restore EXIT

echo "[m4] board=rvv VLEN128 gcc-15.2.0 ATREE=$ATREE core=$CORE loadavg=$(cat /proc/loadavg)"

echo "=== [0] baseline verify (ARCH+GEN) + kernel fingerprints ==="
G0=$(md5sum "$GEN"|awk '{print $1}'); A0=$(md5sum "$ARCH"|awk '{print $1}')
echo "  GEN  now=$G0  base=$BASE_GEN"; echo "  ARCH now=$A0  base=$BASE_ARCH"
if [ "$A0" != "$BASE_ARCH" ] || [ "$G0" != "$BASE_GEN" ]; then echo "  *** A-tree NOT at baseline -- ABORT ***"; exit 10; fi
cp "$ARCH" "$SCR/arch_riscv_repack.cpp.ORIG"; echo "  ARCH backup captured."
echo "  fresh_q4K.inc md5 (expect 90d454da...):"; md5sum "$SCR/fresh_q4K.inc"
echo "  fresh_q5K.inc md5 (expect c209226b...):"; md5sum "$SCR/fresh_q5K.inc"

echo "=== [1] instrument q4_K kernel (pure-observation) ==="
python3 "$SCR/instrument_kernel_q4k.py" || { echo "INSTRUMENT FAILED"; exit 11; }
grep -c "$KSYM4" "$SCR/instr_q4K.inc"; grep -c "G3 M4] tile0" "$SCR/instr_q4K.inc"

echo "=== [2] apply M4 patch (non-destructive include+wrappers) ==="
python3 "$SCR/m4_patch.py" || { echo "PATCH FAILED"; exit 12; }

echo "=== [3] rebuild ggml-cpu ==="
if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/m4_patch_build.log 2>&1; then echo "  patch build OK"
else echo "  patch build FAIL"; tail -50 /tmp/m4_patch_build.log; exit 20; fi
ls -la "$BIN/libggml-cpu.so.0.15.1"
echo "  our kernels present (expect >=1 each):"; nm -C "$BIN/libggml-cpu.so.0.15.1" | grep -c "$KSYM4"; nm -C "$BIN/libggml-cpu.so.0.15.1" | grep -c "$KSYM5"
echo "  wrappers exported (expect 2):"; nm -D "$BIN/libggml-cpu.so.0.15.1" | grep -c "tcrv_m4_call"
echo "  ggml's own gemm still exported (expect >=1 each):"; nm -D "$BIN/libggml-cpu.so.0.15.1" | grep -c "ggml_gemm_q4_K_16x1_q8_K_generic"; nm -D "$BIN/libggml-cpu.so.0.15.1" | grep -c "^.* T ggml_gemm_q4_K_16x1_q8_K$" || nm -D "$BIN/libggml-cpu.so.0.15.1" | grep -cE "ggml_gemm_q4_K_16x1_q8_K$"
echo "  capture globals exported:"; nm -D "$BIN/libggml-cpu.so.0.15.1" | grep -E "tcrv_min_i|tcrv_main_i|tcrv_cap_want" || echo "  (WARN none)"

echo "=== [4] compile m4_probe driver ==="
if g++ -O2 -march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d \
     -I"$INCG" -I"$SCR" "$SCR/m4_probe.cpp" -o "$SCR/m4_probe" \
     -L"$BIN" -Wl,-rpath,"$BIN" -lggml-cpu -lggml-base -lggml -lm 2>/tmp/m4_drv_build.log; then echo "  driver compile OK"
else echo "  driver compile FAIL"; cat /tmp/m4_drv_build.log; exit 30; fi

echo "=== [5] RUN m4_probe (real tensor, dmin!=0, our kernel vs ggml own generic) ==="
LD_LIBRARY_PATH="$BIN" taskset -c "$CORE" "$SCR/m4_probe" "$MODEL"
echo "RUN_EXIT=$?"

echo ""
echo "=== [6] objdump min-fold signature (instruction-level record) ==="
objdump -d "$BIN/libggml-cpu.so.0.15.1" > /tmp/m4_kernel.objdump 2>/dev/null
echo "  vsetivli e32,m2 (8-lane strips):"; grep -cE "vsetivli.*e32,m2|vsetvli.*e32,m2" /tmp/m4_kernel.objdump
echo "  vfnmsac.vv (float dmin*min fold) count:"; grep -cE "vfnmsac\.vv" /tmp/m4_kernel.objdump
echo "objdump saved /tmp/m4_kernel.objdump ($(wc -l </tmp/m4_kernel.objdump) lines)"

echo "=== [7] done -- EXIT trap force-restores ARCH + rebuilds pristine ==="
