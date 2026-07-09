#!/usr/bin/env bash
# [SEL-1 T4b / M2c] Runs ON the board (rvv). Decisive re-emit + re-run: deploy the FRESH
# S6-tiled q4_K repack-GEMM kernel (md5 90d454da) through the SAME reversible M2 dispatch
# patch (was b0b5beac full-unroll golden), rebuild ggml-cpu, run the real ggml_mul_mat
# dispatch driver, compare NORM min-term vs oracle. Forced A-tree restore via EXIT trap.
# NO git stash/rm/mv/add/commit. NO emitter source change. NO local tcrv-opt on board.
set -uo pipefail
ATREE=/home/ubuntu/tcrv-llamacpp
SCR=/tmp/m2c_q4k_dispatch
BUILD=$ATREE/build-gcc15-rv64gcv
BIN=$BUILD/bin
INC=$ATREE/ggml/include
GEN=$ATREE/ggml/src/ggml-cpu/repack.cpp
ARCH=$ATREE/ggml/src/ggml-cpu/arch/riscv/repack.cpp
CORE=${CORE:-8}
BASE_GEN=deb61a29dd079440ffdc8996b5bd2fa1
BASE_ARCH=99131cf791e30348b588423b2388e0b8

source /opt/tcrv-toolchains/env.sh
export LIBRARY_PATH=/opt/tcrv-toolchains/gcc-15.2.0/lib:${LIBRARY_PATH:-}

restore() {
  echo ""
  echo "=== [restore] forced A-tree restore (板末强制 restore) ==="
  if [ -f "$SCR/repack.cpp.ORIG" ] && [ -f "$SCR/arch_riscv_repack.cpp.ORIG" ]; then
    cp "$SCR/repack.cpp.ORIG" "$GEN"
    cp "$SCR/arch_riscv_repack.cpp.ORIG" "$ARCH"
  else
    echo "  (no backups found -- nothing to restore)"; return
  fi
  echo "  restored md5:"; md5sum "$GEN" "$ARCH"
  echo "  expect     : $BASE_GEN  (GEN)"
  echo "  expect     : $BASE_ARCH  (ARCH)"
  echo "=== [restore] rebuild ggml-cpu pristine ==="
  if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/m2c_restore_build.log 2>&1; then
    echo "  restore build OK"
  else
    echo "  restore build FAIL"; tail -25 /tmp/m2c_restore_build.log
  fi
  ls -la "$BIN/libggml-cpu.so.0.15.1" 2>/dev/null
  echo "  golden symbol in pristine .so (expect 0):"; nm -C "$BIN/libggml-cpu.so.0.15.1" 2>/dev/null | grep -c "repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K" || true
  echo "  banner string in pristine .so (expect 0):"; strings "$BIN/libggml-cpu.so.0.15.1" 2>/dev/null | grep -c "TCRV EMITTED GEMM(q4_K_16x1 VLEN128" || true
}
trap restore EXIT

echo "[m2c] board=rvv VLEN128 gcc-15.2.0 ATREE=$ATREE BUILD=build-gcc15-rv64gcv core=$CORE"
echo "[m2c] loadavg=$(cat /proc/loadavg)"
echo ""
echo "=== [0] baseline verify + backup the 2 patch-target files ==="
G0=$(md5sum "$GEN"  | awk '{print $1}')
A0=$(md5sum "$ARCH" | awk '{print $1}')
echo "  GEN  now=$G0  base=$BASE_GEN"
echo "  ARCH now=$A0  base=$BASE_ARCH"
if [ "$G0" != "$BASE_GEN" ] || [ "$A0" != "$BASE_ARCH" ]; then
  echo "  *** A-tree NOT at expected baseline -- ABORT (no patch, no backup) ***"; exit 10
fi
cp "$GEN"  "$SCR/repack.cpp.ORIG"
cp "$ARCH" "$SCR/arch_riscv_repack.cpp.ORIG"
echo "  backups captured (== baseline)."
echo ""
echo "=== [0b] fresh kernel fingerprint on board ==="
md5sum "$SCR/fresh_q4K.inc"
echo "  (expect 90d454da655f2fc1f88435d2d5826942 == S6-tiled fresh; NOT b0b5beac)"
grep -c "tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K" "$SCR/fresh_q4K.inc"
echo ""

echo "=== [1] apply M2c patch (fresh S6 kernel) ==="
python3 "$SCR/m2c_patch.py" || { echo "PATCH FAILED"; exit 11; }
echo ""

echo "=== [2] rebuild ggml-cpu (fresh S6 kernel compiled into libggml-cpu.so) ==="
if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/m2c_patch_build.log 2>&1; then
  echo "  patch build OK"
else
  echo "  patch build FAIL"; tail -40 /tmp/m2c_patch_build.log; exit 20
fi
ls -la "$BIN/libggml-cpu.so.0.15.1"
echo "  golden symbol in patched .so (expect >=1):"; nm -C "$BIN/libggml-cpu.so.0.15.1" | grep -c "repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K"
echo "  banner string in patched .so (expect >=1):"; strings "$BIN/libggml-cpu.so.0.15.1" | grep -c "TCRV EMITTED GEMM(q4_K_16x1 VLEN128"
echo ""

echo "=== [3] compile m2_dispatch driver (g++-15.2, links patched libggml-cpu.so) ==="
if g++ -O2 -march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d \
     -I"$INC" -I"$SCR" "$SCR/m2_dispatch.cpp" -o "$SCR/m2c_dispatch" \
     -L"$BIN" -Wl,-rpath,"$BIN" -lggml-cpu -lggml-base -lggml -lm 2>/tmp/m2c_drv_build.log; then
  echo "  driver compile OK"
else
  echo "  driver compile FAIL"; cat /tmp/m2c_drv_build.log; exit 30
fi
echo "  driver links:"; ldd "$SCR/m2c_dispatch" | grep -E "ggml-cpu|ggml-base|ggml\.so" || true
echo ""

echo "=== [4] RUN m2c_dispatch (real ggml_mul_mat dispatch -> FRESH S6 kernel, min-term exercised) ==="
LD_LIBRARY_PATH="$BIN" taskset -c "$CORE" "$SCR/m2c_dispatch"
echo "RUN_EXIT=$?"
echo ""
echo "=== [5] done -- EXIT trap will now force-restore the A-tree + rebuild pristine ==="
