#!/usr/bin/env bash
# [G5-M2 q6_K] Build OFF (pristine: NO q6_K riscv scaffold -> stock ggml q6_K block-dot)
# + ON (NET-NEW 1,16 riscv scaffold + reused upstream q8_K activation + our EMITTED vl=8
# q6_K repack kernels) .so variants from the SAME tcrv tree (gcc-15.2.0 symmetric; only
# the q6_K scaffold differs). Seal: nm q6_K tcrv symbols present (ON)/absent (OFF),
# banners present, objdump vl-width (must be 8, NEVER 16/64). Then RESTORE 3 source files
# to baseline + rebuild pristine. q6_K edits 3 files (repack.h + GEN + ARCH repack.cpp).
set -uo pipefail
ATREE=/home/ubuntu/tcrv-llamacpp
BUILD=$ATREE/build-gcc15-rv64gcv
BIN=$BUILD/bin
GEN=$ATREE/ggml/src/ggml-cpu/repack.cpp
HDR=$ATREE/ggml/src/ggml-cpu/repack.h
ARCH=$ATREE/ggml/src/ggml-cpu/arch/riscv/repack.cpp
ARCHDIR=$ATREE/ggml/src/ggml-cpu/arch/riscv
SCR=/tmp/g5_q6k
LIVE=$BIN/libggml-cpu.so.0.15.1
BASE_GEN=deb61a29dd079440ffdc8996b5bd2fa1
BASE_HDR=57851439e7c6f5e35aca7986e148e42b
BASE_ARCH=99131cf791e30348b588423b2388e0b8
GEVM_SYM=tcrv_emitc_ggml_repack_gemv_q6_K_q8_K_kernel_ggml_repack_gemv_q6_K_q8_K
GEMM_SYM=tcrv_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K
source /opt/tcrv-toolchains/env.sh
export LIBRARY_PATH=/opt/tcrv-toolchains/gcc-15.2.0/lib:${LIBRARY_PATH:-}
mkdir -p "$SCR"

echo "[g5-m2-q6_K] board=rvv openEuler VLEN128 load=$(cat /proc/loadavg)"
echo "=== [0] baseline verify (3 files) + backup ==="
G0=$(md5sum "$GEN"|awk '{print $1}'); H0=$(md5sum "$HDR"|awk '{print $1}'); A0=$(md5sum "$ARCH"|awk '{print $1}')
echo "  GEN =$G0 (base $BASE_GEN)"; echo "  HDR =$H0 (base $BASE_HDR)"; echo "  ARCH=$A0 (base $BASE_ARCH)"
if [ "$G0" != "$BASE_GEN" ] || [ "$H0" != "$BASE_HDR" ] || [ "$A0" != "$BASE_ARCH" ]; then echo "  *** NOT baseline -- ABORT"; exit 10; fi
cp "$GEN" "$SCR/repack.cpp.ORIG"; cp "$HDR" "$SCR/repack.h.ORIG"; cp "$ARCH" "$SCR/arch_riscv_repack.cpp.ORIG"
echo "  .inc md5s: gemm=$(md5sum "$SCR/tcrv_emitted_gemm_q6_K.inc"|awk '{print $1}') gevm=$(md5sum "$SCR/tcrv_emitted_gevm_q6_K.inc"|awk '{print $1}')"

echo "=== [1] build OFF variant (pristine baseline = NO q6_K riscv scaffold) ==="
touch "$GEN" "$HDR" "$ARCH"
if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/g5_q6k_build_off.log 2>&1; then echo "  OFF build OK"
else echo "  OFF build FAIL"; tail -40 /tmp/g5_q6k_build_off.log; exit 20; fi
cp -f "$LIVE" "$SCR/libggml-cpu.so.q6OFF"
OFFMD5=$(md5sum "$SCR/libggml-cpu.so.q6OFF"|awk '{print $1}')
OFFSYM=$(nm -C "$SCR/libggml-cpu.so.q6OFF" 2>/dev/null | grep -cE "tcrv_emitc_ggml_repack_(gemv|gemm)_q6_K" || true)
echo "  OFF .so md5=$OFFMD5  q6_K_tcrv_sym(expect 0)=$OFFSYM"

echo "=== [2] copy emitted .inc + apply scaffold patch + build ON ==="
cp -f "$SCR/tcrv_emitted_gemm_q6_K.inc" "$ARCHDIR/tcrv_emitted_gemm_q6_K.inc"
cp -f "$SCR/tcrv_emitted_gevm_q6_K.inc" "$ARCHDIR/tcrv_emitted_gevm_q6_K.inc"
python3 "$SCR/deploy_patch_q6_K_emitted.py" || { echo "PATCH FAILED"; exit 11; }
touch "$GEN" "$HDR" "$ARCH"
if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/g5_q6k_build_on.log 2>&1; then echo "  ON build OK"
else echo "  ON build FAIL"; tail -80 /tmp/g5_q6k_build_on.log; exit 21; fi
cp -f "$LIVE" "$SCR/libggml-cpu.so.q6ON"
ONMD5=$(md5sum "$SCR/libggml-cpu.so.q6ON"|awk '{print $1}')
echo "  ON .so md5=$ONMD5"

echo "=== [3] SEAL: nm symbols + banners + objdump vl-width ==="
nm -C "$SCR/libggml-cpu.so.q6ON" 2>/dev/null | grep -E "tcrv_emitc_ggml_repack_(gemv|gemm)_q6_K" | sed 's/^/    /'
ONGEVM=$(strings "$SCR/libggml-cpu.so.q6ON" | grep -c "TCRV G5-M2 EMITTED GEVM(q6_K_16x1 VLEN128" || true)
ONGEMM=$(strings "$SCR/libggml-cpu.so.q6ON" | grep -c "TCRV G5-M2 EMITTED GEMM(q6_K_16x1 VLEN128" || true)
echo "  banner gevm(expect>=1)=$ONGEVM  gemm(expect>=1)=$ONGEMM"
[ "$ONMD5" != "$OFFMD5" ] && echo "  ON != OFF (variants differ, good)" || echo "  *** ON == OFF -- patch had no effect!"
echo "  -- objdump vsetivli widths inside emitted q6_K symbols (must be 8, NEVER 16/64) --"
for sym in "$GEVM_SYM" "$GEMM_SYM"; do
  objdump -d --disassemble="$sym" "$SCR/libggml-cpu.so.q6ON" 2>/dev/null > "$SCR/objdump_$sym.txt"
  N8=$(grep -cE 'vsetivli[^,]*,[[:space:]]*8,' "$SCR/objdump_$sym.txt" || true)
  N16=$(grep -cE 'vsetivli[^,]*,[[:space:]]*16,' "$SCR/objdump_$sym.txt" || true)
  N64=$(grep -cE 'vsetivli[^,]*,[[:space:]]*64,' "$SCR/objdump_$sym.txt" || true)
  NDyn=$(grep -cE 'vsetvli' "$SCR/objdump_$sym.txt" || true)
  VForms=$(grep -oE 'vset[i]*vli[^#]*' "$SCR/objdump_$sym.txt" | grep -oE 'e(8|16|32),(mf2|mf4|m1|m2|m4)' | sort | uniq -c | tr '\n' ' ')
  echo "    $sym:"
  echo "      vsetivli(imm=8)=$N8  vsetivli(imm=16)=$N16[MUST BE 0]  vsetivli(imm=64)=$N64[MUST BE 0]  dyn_vsetvli=$NDyn"
  echo "      SEW,LMUL forms: $VForms"
done

echo "=== [4] restore 3 source files to baseline + rebuild pristine ==="
cp "$SCR/repack.cpp.ORIG" "$GEN"; cp "$SCR/repack.h.ORIG" "$HDR"; cp "$SCR/arch_riscv_repack.cpp.ORIG" "$ARCH"
rm -f "$ARCHDIR/tcrv_emitted_gemm_q6_K.inc" "$ARCHDIR/tcrv_emitted_gevm_q6_K.inc"
GR=$(md5sum "$GEN"|awk '{print $1}'); HR=$(md5sum "$HDR"|awk '{print $1}'); AR=$(md5sum "$ARCH"|awk '{print $1}')
echo "  restored md5: GEN=$GR HDR=$HR ARCH=$AR"
[ "$GR" = "$BASE_GEN" ] && [ "$HR" = "$BASE_HDR" ] && [ "$AR" = "$BASE_ARCH" ] && echo "  SOURCE RESTORED byte-exact" || { echo "  *** restore mismatch"; exit 30; }
touch "$GEN" "$HDR" "$ARCH"
if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/g5_q6k_build_restore.log 2>&1; then echo "  pristine rebuild OK"
else echo "  pristine rebuild FAIL"; tail -25 /tmp/g5_q6k_build_restore.log; exit 31; fi
LIVEMD5=$(md5sum "$LIVE"|awk '{print $1}')
LIVESYM=$(nm -C "$LIVE" 2>/dev/null | grep -cE "tcrv_emitc_ggml_repack_(gemv|gemm)_q6_K" || true)
echo "  live .so md5=$LIVEMD5 (expect==OFF $OFFMD5)  q6_K_sym(expect 0)=$LIVESYM"
[ "$LIVEMD5" = "$OFFMD5" ] && echo "  LIVE == OFF-pristine (zero net change)" || echo "  NOTE: live!=OFF md5 (build nondeterminism); q6_K_sym=$LIVESYM is the seal"
echo "=== [g5-m2-q6_K build+seal DONE] variants:"
md5sum "$SCR/libggml-cpu.so.q6OFF" "$SCR/libggml-cpu.so.q6ON" "$LIVE"
