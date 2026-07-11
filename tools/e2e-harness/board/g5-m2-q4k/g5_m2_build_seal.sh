#!/usr/bin/env bash
# [G5 M2 q4_K] Build OFF (stock: q4_K@VLEN128 gate OFF -> generic path) + ON (our
# CURRENT-HEAD EMITTED vl=8 q4_K repack kernels intercepting VLEN128) .so variants
# from the SAME tcrv tree (gcc-15.2.0 symmetric; only the q4_K gate + emitted-call
# branches + two #include differ). Seal: nm q4_K tcrv symbols present, banners
# present, objdump vl-width (must be 8, NEVER 16). Then RESTORE source to baseline
# + rebuild pristine. Variants saved in $SCR.
set -uo pipefail
ATREE=/home/ubuntu/tcrv-llamacpp
BUILD=$ATREE/build-gcc15-rv64gcv
BIN=$BUILD/bin
GEN=$ATREE/ggml/src/ggml-cpu/repack.cpp
ARCH=$ATREE/ggml/src/ggml-cpu/arch/riscv/repack.cpp
ARCHDIR=$ATREE/ggml/src/ggml-cpu/arch/riscv
SCR=/tmp/g5_q4k
LIVE=$BIN/libggml-cpu.so.0.15.1
BASE_GEN=deb61a29dd079440ffdc8996b5bd2fa1
BASE_ARCH=99131cf791e30348b588423b2388e0b8
source /opt/tcrv-toolchains/env.sh
export LIBRARY_PATH=/opt/tcrv-toolchains/gcc-15.2.0/lib:${LIBRARY_PATH:-}
mkdir -p "$SCR"

echo "[g5-m2-q4k] board=rvv openEuler VLEN128 load=$(cat /proc/loadavg)"
echo "=== [0] baseline verify + backup ==="
G0=$(md5sum "$GEN"|awk '{print $1}'); A0=$(md5sum "$ARCH"|awk '{print $1}')
echo "  GEN  now=$G0 base=$BASE_GEN"; echo "  ARCH now=$A0 base=$BASE_ARCH"
if [ "$G0" != "$BASE_GEN" ] || [ "$A0" != "$BASE_ARCH" ]; then echo "  *** NOT baseline -- ABORT"; exit 10; fi
cp "$GEN" "$SCR/repack.cpp.ORIG"; cp "$ARCH" "$SCR/arch_riscv_repack.cpp.ORIG"
echo "  backups saved in $SCR"
echo "  .inc md5s: gemm=$(md5sum "$SCR/tcrv_emitted_gemm_q4_K.inc"|awk '{print $1}') gevm=$(md5sum "$SCR/tcrv_emitted_gevm_q4_K.inc"|awk '{print $1}')"

echo "=== [1] build OFF variant (baseline source = q4_K gate OFF / generic path) ==="
touch "$GEN" "$ARCH"
if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/g5_q4k_build_off.log 2>&1; then echo "  OFF build OK"
else echo "  OFF build FAIL"; tail -40 /tmp/g5_q4k_build_off.log; exit 20; fi
cp -f "$LIVE" "$SCR/libggml-cpu.so.q4kOFF"
OFFMD5=$(md5sum "$SCR/libggml-cpu.so.q4kOFF"|awk '{print $1}')
OFFSYM=$(nm -C "$SCR/libggml-cpu.so.q4kOFF" 2>/dev/null | grep -c "tcrv_emitc_ggml_repack_gem._q4_K_q8_K_kernel" || true)
echo "  OFF .so md5=$OFFMD5  q4_K_tcrv_sym(expect 0)=$OFFSYM"

echo "=== [2] copy emitted .inc into ARCHDIR + patch ON (emitted vl=8 intercept) + build ==="
cp -f "$SCR/tcrv_emitted_gemm_q4_K.inc" "$ARCHDIR/tcrv_emitted_gemm_q4_K.inc"
cp -f "$SCR/tcrv_emitted_gevm_q4_K.inc" "$ARCHDIR/tcrv_emitted_gevm_q4_K.inc"
python3 "$SCR/deploy_patch_q4k_emitted.py" || { echo "PATCH FAILED"; exit 11; }
touch "$GEN" "$ARCH"
if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/g5_q4k_build_on.log 2>&1; then echo "  ON build OK"
else echo "  ON build FAIL"; tail -60 /tmp/g5_q4k_build_on.log; exit 21; fi
cp -f "$LIVE" "$SCR/libggml-cpu.so.q4kON"
ONMD5=$(md5sum "$SCR/libggml-cpu.so.q4kON"|awk '{print $1}')
echo "  ON .so md5=$ONMD5"

echo "=== [3] SEAL: nm symbols + banners + objdump vl-width ==="
echo "  -- nm q4_K tcrv symbols (expect gevm + gemm) --"
nm -C "$SCR/libggml-cpu.so.q4kON" 2>/dev/null | grep "tcrv_emitc_ggml_repack_gem._q4_K_q8_K_kernel" | sed 's/^/    /'
ONGEVM=$(strings "$SCR/libggml-cpu.so.q4kON" | grep -c "TCRV G5-M2 EMITTED GEVM(q4_K_16x1 VLEN128" || true)
ONGEMM=$(strings "$SCR/libggml-cpu.so.q4kON" | grep -c "TCRV G5-M2 EMITTED GEMM(q4_K_16x1 VLEN128" || true)
echo "  banner gevm(expect>=1)=$ONGEVM  gemm(expect>=1)=$ONGEMM"
[ "$ONMD5" != "$OFFMD5" ] && echo "  ON != OFF (variants differ, good)" || echo "  *** ON == OFF -- patch had no effect!"
echo "  -- objdump vsetivli widths inside emitted q4_K symbols (must be 8, NEVER 16) --"
for sym in tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K; do
  objdump -d --disassemble="$sym" "$SCR/libggml-cpu.so.q4kON" 2>/dev/null > "$SCR/objdump_$sym.txt"
  N8=$(grep -cE 'vsetivli[^,]*,[[:space:]]*8,' "$SCR/objdump_$sym.txt" || true)
  N16=$(grep -cE 'vsetivli[^,]*,[[:space:]]*16,' "$SCR/objdump_$sym.txt" || true)
  N64=$(grep -cE 'vsetivli[^,]*,[[:space:]]*64,' "$SCR/objdump_$sym.txt" || true)
  VForms=$(grep -oE 'vset[i]*vli[^#]*' "$SCR/objdump_$sym.txt" | grep -oE 'e(8|16|32),(mf2|mf4|m1|m2|m4)' | sort | uniq -c | tr '\n' ' ')
  echo "    $sym:"
  echo "      vsetivli(imm=8)=$N8  vsetivli(imm=16)=$N16[MUST BE 0]  vsetivli(imm=64)=$N64[MUST BE 0]"
  echo "      SEW,LMUL forms: $VForms"
done

echo "=== [4] restore source to baseline + rebuild pristine ==="
cp "$SCR/repack.cpp.ORIG" "$GEN"; cp "$SCR/arch_riscv_repack.cpp.ORIG" "$ARCH"
rm -f "$ARCHDIR/tcrv_emitted_gemm_q4_K.inc" "$ARCHDIR/tcrv_emitted_gevm_q4_K.inc"
GR=$(md5sum "$GEN"|awk '{print $1}'); AR=$(md5sum "$ARCH"|awk '{print $1}')
echo "  restored source md5: GEN=$GR ARCH=$AR"
[ "$GR" = "$BASE_GEN" ] && [ "$AR" = "$BASE_ARCH" ] && echo "  SOURCE RESTORED byte-exact" || { echo "  *** restore mismatch"; exit 30; }
touch "$GEN" "$ARCH"
if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/g5_q4k_build_restore.log 2>&1; then echo "  pristine rebuild OK"
else echo "  pristine rebuild FAIL"; tail -25 /tmp/g5_q4k_build_restore.log; exit 31; fi
LIVEMD5=$(md5sum "$LIVE"|awk '{print $1}')
LIVESYM=$(nm -C "$LIVE" 2>/dev/null | grep -c "q4_K_q8_K_kernel" || true)
echo "  live .so md5=$LIVEMD5 (expect==OFF $OFFMD5)  q4_K_sym(expect 0)=$LIVESYM"
[ "$LIVEMD5" = "$OFFMD5" ] && echo "  LIVE == OFF-pristine (zero net change)" || echo "  NOTE: live!=OFF md5 (build nondeterminism); q4_K_sym=$LIVESYM is the seal"
echo "=== [g5-m2-q4k build+seal DONE] variants:"
md5sum "$SCR/libggml-cpu.so.q4kOFF" "$SCR/libggml-cpu.so.q4kON" "$LIVE"
