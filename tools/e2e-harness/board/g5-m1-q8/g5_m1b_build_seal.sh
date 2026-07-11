#!/usr/bin/env bash
# [G5 M1b] Build OFF (stock block-dot) + ON (our EMITTED vl=8 q8_0 repack kernels
# intercepting VLEN128) .so variants from the SAME tcrv tree (gcc-15.2.0 symmetric;
# only the q8_0 gate + emitted-call branches + one #include differ). Seal: nm q8_0
# tcrv symbols present, banners present, objdump vl-width (must be 8, never 16).
# Then RESTORE source to baseline + rebuild pristine. Variants saved in $SCR.
set -uo pipefail
ATREE=/home/ubuntu/tcrv-llamacpp
BUILD=$ATREE/build-gcc15-rv64gcv
BIN=$BUILD/bin
GEN=$ATREE/ggml/src/ggml-cpu/repack.cpp
ARCH=$ATREE/ggml/src/ggml-cpu/arch/riscv/repack.cpp
ARCHDIR=$ATREE/ggml/src/ggml-cpu/arch/riscv
SCR=/tmp/g5_q8b
LIVE=$BIN/libggml-cpu.so.0.15.1
BASE_GEN=deb61a29dd079440ffdc8996b5bd2fa1
BASE_ARCH=99131cf791e30348b588423b2388e0b8
source /opt/tcrv-toolchains/env.sh
export LIBRARY_PATH=/opt/tcrv-toolchains/gcc-15.2.0/lib:${LIBRARY_PATH:-}
mkdir -p "$SCR"

echo "[g5-m1b] board=rvv openEuler VLEN128 load=$(cat /proc/loadavg)"
echo "=== [0] baseline verify + backup ==="
G0=$(md5sum "$GEN"|awk '{print $1}'); A0=$(md5sum "$ARCH"|awk '{print $1}')
echo "  GEN  now=$G0 base=$BASE_GEN"; echo "  ARCH now=$A0 base=$BASE_ARCH"
if [ "$G0" != "$BASE_GEN" ] || [ "$A0" != "$BASE_ARCH" ]; then echo "  *** NOT baseline -- ABORT"; exit 10; fi
cp "$GEN" "$SCR/repack.cpp.ORIG"; cp "$ARCH" "$SCR/arch_riscv_repack.cpp.ORIG"
echo "  backups saved in $SCR"

echo "=== [1] build OFF variant (baseline source = q8_0 gate OFF / block-dot) ==="
touch "$GEN" "$ARCH"
if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/g5b_build_off.log 2>&1; then echo "  OFF build OK"
else echo "  OFF build FAIL"; tail -40 /tmp/g5b_build_off.log; exit 20; fi
cp -f "$LIVE" "$SCR/libggml-cpu.so.q8OFF"
OFFMD5=$(md5sum "$SCR/libggml-cpu.so.q8OFF"|awk '{print $1}')
OFFSYM=$(nm -C "$SCR/libggml-cpu.so.q8OFF" 2>/dev/null | grep -c "tcrv_emitc_ggml_.*q8_0_q8_0_kernel" || true)
echo "  OFF .so md5=$OFFMD5  q8_0_tcrv_sym(expect 0)=$OFFSYM"

echo "=== [2] copy emitted .inc + patch ON (emitted vl=8 intercept) + build ==="
cp -f "$SCR/tcrv_emitted_q8_0.inc" "$ARCHDIR/tcrv_emitted_q8_0.inc"
echo "  .inc md5=$(md5sum "$ARCHDIR/tcrv_emitted_q8_0.inc"|awk '{print $1}')"
python3 "$SCR/deploy_patch_q8_emitted.py" || { echo "PATCH FAILED"; exit 11; }
touch "$GEN" "$ARCH"
if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/g5b_build_on.log 2>&1; then echo "  ON build OK"
else echo "  ON build FAIL"; tail -50 /tmp/g5b_build_on.log; exit 21; fi
cp -f "$LIVE" "$SCR/libggml-cpu.so.q8ON"
ONMD5=$(md5sum "$SCR/libggml-cpu.so.q8ON"|awk '{print $1}')
echo "  ON .so md5=$ONMD5"

echo "=== [3] SEAL: nm symbols + banners + objdump vl-width ==="
echo "  -- nm q8_0 tcrv symbols (expect gevm + gemm) --"
nm -C "$SCR/libggml-cpu.so.q8ON" 2>/dev/null | grep "tcrv_emitc_ggml_.*q8_0_q8_0_kernel" | sed 's/^/    /'
ONGEVM=$(strings "$SCR/libggml-cpu.so.q8ON" | grep -c "TCRV G5-M1b EMITTED GEVM(q8_0_16x1 VLEN128" || true)
ONGEMM=$(strings "$SCR/libggml-cpu.so.q8ON" | grep -c "TCRV G5-M1b EMITTED GEMM(q8_0_16x1 VLEN128" || true)
echo "  banner gevm(expect>=1)=$ONGEVM  gemm(expect>=1)=$ONGEMM"
[ "$ONMD5" != "$OFFMD5" ] && echo "  ON != OFF (variants differ, good)" || echo "  *** ON == OFF -- patch had no effect!"
echo "  -- objdump vsetivli widths inside emitted q8_0 symbols (must be 8, NEVER 16) --"
for sym in tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0 tcrv_emitc_ggml_gemm_q8_0_q8_0_kernel_ggml_gemm_q8_0_q8_0; do
  objdump -d --disassemble="$sym" "$SCR/libggml-cpu.so.q8ON" 2>/dev/null > "$SCR/objdump_$sym.txt"
  W=$(grep -oE 'vset[i]*vli[^#]*' "$SCR/objdump_$sym.txt" | grep -oE 'e(8|16|32),(mf2|m1|m2)' | sort | uniq -c)
  N8=$(grep -cE 'vsetivli[^,]*,8,' "$SCR/objdump_$sym.txt" || true)
  N16=$(grep -cE 'vsetivli[^,]*,16,' "$SCR/objdump_$sym.txt" || true)
  echo "    $sym: vsetivli(imm=8)=$N8  vsetivli(imm=16)=$N16"
done

echo "=== [4] restore source to baseline + rebuild pristine ==="
cp "$SCR/repack.cpp.ORIG" "$GEN"; cp "$SCR/arch_riscv_repack.cpp.ORIG" "$ARCH"
rm -f "$ARCHDIR/tcrv_emitted_q8_0.inc"
GR=$(md5sum "$GEN"|awk '{print $1}'); AR=$(md5sum "$ARCH"|awk '{print $1}')
echo "  restored source md5: GEN=$GR ARCH=$AR"
[ "$GR" = "$BASE_GEN" ] && [ "$AR" = "$BASE_ARCH" ] && echo "  SOURCE RESTORED byte-exact" || { echo "  *** restore mismatch"; exit 30; }
touch "$GEN" "$ARCH"
if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/g5b_build_restore.log 2>&1; then echo "  pristine rebuild OK"
else echo "  pristine rebuild FAIL"; tail -25 /tmp/g5b_build_restore.log; exit 31; fi
LIVEMD5=$(md5sum "$LIVE"|awk '{print $1}')
LIVESYM=$(nm -C "$LIVE" 2>/dev/null | grep -c "q8_0_q8_0_kernel" || true)
echo "  live .so md5=$LIVEMD5 (expect==OFF $OFFMD5)  q8_0_sym(expect 0)=$LIVESYM"
[ "$LIVEMD5" = "$OFFMD5" ] && echo "  LIVE == OFF-pristine (zero net change)" || echo "  NOTE: live!=OFF md5 (build nondeterminism); q8_0_sym=$LIVESYM is the seal"
echo "=== [g5-m1b build+seal DONE] variants:"
md5sum "$SCR/libggml-cpu.so.q8OFF" "$SCR/libggml-cpu.so.q8ON" "$LIVE"
