#!/usr/bin/env bash
# [G3 T4b seal-fix] Board restore: restore the 2 A-tree source files to baseline
# (GEN deb61a29 / ARCH 99131cf7) + rebuild pristine ggml-cpu. Leaves the FIXED .A-q4kON
# in place (deliverable) with its vl16 backup at $SCR/A-q4kON.vl16-garbage.bak.
set -uo pipefail
ATREE=/home/ubuntu/tcrv-llamacpp
BUILD=$ATREE/build-gcc15-rv64gcv
BIN=$BUILD/bin
GEN=$ATREE/ggml/src/ggml-cpu/repack.cpp
ARCH=$ATREE/ggml/src/ggml-cpu/arch/riscv/repack.cpp
SCR=/tmp/t4b_seal_fix
BASE_GEN=deb61a29dd079440ffdc8996b5bd2fa1
BASE_ARCH=99131cf791e30348b588423b2388e0b8
GSYM=tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K
source /opt/tcrv-toolchains/env.sh
export LIBRARY_PATH=/opt/tcrv-toolchains/gcc-15.2.0/lib:${LIBRARY_PATH:-}
echo "=== [restore] source files ==="
if [ -f "$SCR/repack.cpp.ORIG" ] && [ -f "$SCR/arch_riscv_repack.cpp.ORIG" ]; then
  cp "$SCR/repack.cpp.ORIG" "$GEN"; cp "$SCR/arch_riscv_repack.cpp.ORIG" "$ARCH"
else echo "  (no backups -- nothing to restore)"; exit 1; fi
echo "  restored md5:"; md5sum "$GEN" "$ARCH"
echo "  expect GEN=$BASE_GEN ARCH=$BASE_ARCH"
G0=$(md5sum "$GEN"|awk '{print $1}'); A0=$(md5sum "$ARCH"|awk '{print $1}')
[ "$G0" = "$BASE_GEN" ] && [ "$A0" = "$BASE_ARCH" ] && echo "  SOURCE RESTORED byte-exact" || echo "  *** restore mismatch"
echo "=== [restore] rebuild pristine ggml-cpu ==="
if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/seal_restore_build.log 2>&1; then echo "  restore build OK"
else echo "  restore build FAIL"; tail -25 /tmp/seal_restore_build.log; fi
echo "  q4_K gemm sym in pristine live .so (expect 0): $(nm -C "$BIN/libggml-cpu.so.0.15.1" 2>/dev/null | grep -c "$GSYM")"
echo "  fixed .A-q4kON retained:"; ls -la "$BIN/libggml-cpu.so.0.15.1.A-q4kON" 2>/dev/null
echo "=== [restore done] ==="
