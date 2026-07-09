#!/usr/bin/env bash
# [G3 T4b seal-fix] Board (rvv, VLEN128). Backup 2 A-tree files -> patch (clean q4_K gemm+gevm
# VLEN128 vl=8) -> rebuild ggml-cpu -> objdump vl=8 seal (gemm+gevm) -> save fixed .so to .A-q4kON
# (old vl16 build backed up). Leaves tree PATCHED (correctness+perf runs follow in-place).
# Restore is a SEPARATE step (board_restore.sh). NO git stash/rm/mv/add/commit.
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
VSYM=tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K
source /opt/tcrv-toolchains/env.sh
export LIBRARY_PATH=/opt/tcrv-toolchains/gcc-15.2.0/lib:${LIBRARY_PATH:-}

echo "[deploy] board=rvv VLEN128 ATREE=$ATREE load=$(cat /proc/loadavg)"
echo "=== [0] baseline verify + backup ==="
G0=$(md5sum "$GEN"|awk '{print $1}'); A0=$(md5sum "$ARCH"|awk '{print $1}')
echo "  GEN  now=$G0 base=$BASE_GEN"; echo "  ARCH now=$A0 base=$BASE_ARCH"
if [ "$G0" != "$BASE_GEN" ] || [ "$A0" != "$BASE_ARCH" ]; then echo "  *** NOT baseline -- ABORT"; exit 10; fi
cp "$GEN" "$SCR/repack.cpp.ORIG"; cp "$ARCH" "$SCR/arch_riscv_repack.cpp.ORIG"
echo "  backups: $SCR/{repack.cpp.ORIG,arch_riscv_repack.cpp.ORIG}"
echo "=== [0b] kernel fingerprints ==="
md5sum "$SCR/gemm_q4_K.inc" "$SCR/gemv_q4_K.inc"
echo "  gemm sym in inc: $(grep -c "$GSYM" "$SCR/gemm_q4_K.inc")  gevm sym in inc: $(grep -c "$VSYM" "$SCR/gemv_q4_K.inc")"
echo "  q4_K already in q4_0 incs (expect 0): $(grep -c q4_K "$ARCH".dummy 2>/dev/null || grep -lc q4_K $ATREE/ggml/src/ggml-cpu/arch/riscv/tcrv_emitted_repack_gem*.inc 2>/dev/null | wc -l)"

echo "=== [1] apply seal-fix patch ==="
python3 "$SCR/deploy_patch.py" || { echo "PATCH FAILED"; exit 11; }

echo "=== [2] rebuild ggml-cpu ==="
if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/seal_patch_build.log 2>&1; then echo "  build OK"
else echo "  build FAIL"; tail -50 /tmp/seal_patch_build.log; exit 20; fi
ls -la "$BIN/libggml-cpu.so.0.15.1"
echo "  gemm sym in .so (expect >=1): $(nm -C "$BIN/libggml-cpu.so.0.15.1" | grep -c "$GSYM")"
echo "  gevm sym in .so (expect >=1): $(nm -C "$BIN/libggml-cpu.so.0.15.1" | grep -c "$VSYM")"
echo "  gemm banner (expect >=1): $(strings "$BIN/libggml-cpu.so.0.15.1" | grep -c "TCRV EMITTED GEMM(q4_K_16x1 VLEN128")"
echo "  gevm banner (expect >=1): $(strings "$BIN/libggml-cpu.so.0.15.1" | grep -c "TCRV EMITTED GEMV(q4_K_16x1 VLEN128")"

echo "=== [3] objdump vl=8 seal (gemm+gevm) ==="
objdump -d "$BIN/libggml-cpu.so.0.15.1" > /tmp/seal_kernel.objdump 2>/dev/null
for SYM in "$GSYM" "$VSYM"; do
  echo "  --- $SYM ---"
  START=$(grep -nE "<$SYM>:" /tmp/seal_kernel.objdump | head -1 | cut -d: -f1)
  if [ -z "$START" ]; then echo "    (symbol not in objdump -- may be inlined)"; continue; fi
  END=$(awk -v s="$START" 'NR>s && /^$/ {print NR; exit}' /tmp/seal_kernel.objdump)
  SEG=$(sed -n "${START},${END}p" /tmp/seal_kernel.objdump)
  echo "    vsetivli ...,8,e32,m2 (vl=8, expect >0): $(echo "$SEG" | grep -cE "vsetivli\s+[a-z0-9]+,8,e32,m2")"
  echo "    vsetivli ...,16,e32,m2 (vl=16, expect 0):  $(echo "$SEG" | grep -cE "vsetivli\s+[a-z0-9]+,16,e32,m2")"
  echo "    any e32,m2 vset lines (sample):"; echo "$SEG" | grep -oE "vset[a-z]+\s+[a-z0-9]+,[0-9]+,e32,m2" | sort | uniq -c | head
done

echo "=== [4] save fixed build to .A-q4kON (backup old vl16) ==="
cp -f "$BIN/libggml-cpu.so.0.15.1.A-q4kON" "$SCR/A-q4kON.vl16-garbage.bak" 2>/dev/null && echo "  old .A-q4kON -> $SCR/A-q4kON.vl16-garbage.bak"
cp -f "$BIN/libggml-cpu.so.0.15.1" "$BIN/libggml-cpu.so.0.15.1.A-q4kON"
echo "  fixed build saved to .A-q4kON:"; md5sum "$BIN/libggml-cpu.so.0.15.1" "$BIN/libggml-cpu.so.0.15.1.A-q4kON"
echo "=== [deploy done -- tree PATCHED, .A-q4kON=fixed; run correctness next] ==="
