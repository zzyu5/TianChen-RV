#!/usr/bin/env bash
# [decisive-kquant-gcc-vs-vlen] SURGICAL compiler-isolation build.
# Produces 3 libggml-cpu.so variants (all from the gcc-15 build tree; block-dot etc.
# stay gcc-15 = T-PERF1b account):
#   q4kOFF.gcc          = baseline block-dot (gate OFF)                  [DENOM both accounts]
#   q4kON.gcc           = emitted q4_K repack, gcc-15 compiled (742 spill) [gcc account NUMER]
#   q4kON.clangrepack   = emitted q4_K repack, CLANG-18 compiled (4 spill) [clang account NUMER]
# The ONLY difference q4kON.gcc vs q4kON.clangrepack = compiler of the 2 repack TUs
# (ggml-cpu/repack.cpp GEN + arch/riscv/repack.cpp ARCH, the latter holding the emitted
# kernel). Everything else identical (same gcc objects, same g++ -shared link). This
# isolates gcc-742-spill-death from VLEN128-recon on the EXACT deployed kernel. NO git.
set -uo pipefail
ATREE=/home/ubuntu/tcrv-llamacpp
B=$ATREE/build-gcc15-rv64gcv
BIN=$B/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
GEN=$ATREE/ggml/src/ggml-cpu/repack.cpp
ARCH=$ATREE/ggml/src/ggml-cpu/arch/riscv/repack.cpp
ARCHDIR=$ATREE/ggml/src/ggml-cpu/arch/riscv
SCR=/tmp/dkgv
BASE_GEN=deb61a29dd079440ffdc8996b5bd2fa1
BASE_ARCH=99131cf791e30348b588423b2388e0b8
CLXX=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang++
GCCRT=/usr/lib/gcc/riscv64-openEuler-linux/12
FULLM="rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause"
REDM="rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zicbop_zihintpause"
ISYS="-isystem /usr/include/c++/12 -isystem /usr/include/c++/12/riscv64-openEuler-linux -isystem /usr/include/c++/12/backward"
OARCH=ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/riscv/repack.cpp.o
OGEN=ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/repack.cpp.o
source /opt/tcrv-toolchains/env.sh
export LD_LIBRARY_PATH="$BIN:/opt/tcrv-toolchains/gcc-15.2.0/lib:${LD_LIBRARY_PATH:-}"

echo "[dkgv-mixed] load=$(cat /proc/loadavg)"
echo "=== [0] baseline verify + backup ==="
G0=$(md5sum "$GEN"|awk '{print $1}'); A0=$(md5sum "$ARCH"|awk '{print $1}')
[ "$G0" = "$BASE_GEN" ] && [ "$A0" = "$BASE_ARCH" ] || { echo "*** NOT baseline GEN=$G0 ARCH=$A0"; exit 10; }
cp "$GEN" "$SCR/GEN.ORIG"; cp "$ARCH" "$SCR/ARCH.ORIG"
# safety: restore source + remove stray .inc on ANY exit (idempotent w/ step 6)
trap 'cp -f "$SCR/GEN.ORIG" "$GEN" 2>/dev/null; cp -f "$SCR/ARCH.ORIG" "$ARCH" 2>/dev/null; rm -f "$ARCHDIR/tcrv_emitted_gemm_q4_K.inc" "$ARCHDIR/tcrv_emitted_gevm_q4_K.inc" 2>/dev/null' EXIT

echo "=== [1] gcc pristine build -> q4kOFF.gcc (block-dot baseline) ==="
touch "$GEN" "$ARCH"; ninja -C "$B" ggml-cpu > "$SCR/mb_off.log" 2>&1 || { echo "OFF FAIL"; tail -20 "$SCR/mb_off.log"; exit 20; }
cp -f "$LIVE" "$SCR/libggml-cpu.so.q4kOFF.gcc"
echo "  q4kOFF.gcc md5=$(md5sum "$SCR/libggml-cpu.so.q4kOFF.gcc"|cut -c1-12) tcrv_sym=$(nm -C "$SCR/libggml-cpu.so.q4kOFF.gcc"|grep -c q4_K_q8_K_kernel||true)"

echo "=== [2] patch source (emitted vl=8 intercept) + gcc build -> q4kON.gcc ==="
cp -f "$SCR/tcrv_emitted_gemm_q4_K.inc" "$ARCHDIR/"; cp -f "$SCR/tcrv_emitted_gevm_q4_K.inc" "$ARCHDIR/"
python3 "$SCR/deploy_patch_q4k_emitted.py" || { echo "PATCH FAIL"; exit 11; }
touch "$GEN" "$ARCH"; ninja -C "$B" ggml-cpu > "$SCR/mb_on_gcc.log" 2>&1 || { echo "ON-GCC FAIL"; tail -30 "$SCR/mb_on_gcc.log"; exit 21; }
cp -f "$LIVE" "$SCR/libggml-cpu.so.q4kON.gcc"
echo "  q4kON.gcc md5=$(md5sum "$SCR/libggml-cpu.so.q4kON.gcc"|cut -c1-12)"

echo "=== [3] recompile the 2 repack TUs with CLANG-18 (patched source), overwrite .o ==="
mkgen(){  # $1 = captured gcc cmd file ; transform -> clang
  sed -e 's#^ccache ##' \
      -e "s#/opt/tcrv-toolchains/gcc-15.2.0/bin/g++#$CLXX $ISYS#" \
      -e "s#$FULLM#$REDM#g" "$1"
}
cd "$B"
echo "  -- clang arch/riscv/repack.cpp.o --"; mkgen "$SCR/cmd_arch_repack.txt" > "$SCR/run_arch.sh"; bash "$SCR/run_arch.sh" 2>"$SCR/clang_arch.err" && echo "    OK $(stat -c%s $OARCH)B" || { echo "    ARCH CLANG FAIL"; head -12 "$SCR/clang_arch.err"; exit 22; }
echo "  -- clang ggml-cpu/repack.cpp.o (GEN) --"; mkgen "$SCR/cmd_gen_repack.txt" > "$SCR/run_gen.sh"; bash "$SCR/run_gen.sh" 2>"$SCR/clang_gen.err" && echo "    OK $(stat -c%s $OGEN)B" || { echo "    GEN CLANG FAIL"; head -12 "$SCR/clang_gen.err"; exit 23; }

echo "=== [4] relink (g++ -shared) with clang repack .o -> q4kON.clangrepack ==="
bash "$SCR/cmd_link.txt" 2>"$SCR/relink.err" && echo "  relink OK" || { echo "  RELINK FAIL"; head -20 "$SCR/relink.err"; exit 24; }
cp -f "$LIVE" "$SCR/libggml-cpu.so.q4kON.clangrepack"
echo "  q4kON.clangrepack md5=$(md5sum "$SCR/libggml-cpu.so.q4kON.clangrepack"|cut -c1-12)"

echo "=== [5] SEAL: nm + banner + objdump spill (gcc vs clangrepack) ==="
for V in q4kON.gcc q4kON.clangrepack; do
  L="$SCR/libggml-cpu.so.$V"
  echo "  [$V] banner_gemm=$(strings "$L"|grep -c 'TCRV G5-M2 EMITTED GEMM(q4_K_16x1 VLEN128') tcrv_sym=$(nm -C "$L"|grep -c q4_K_q8_K_kernel||true)"
  for sym in gemv gemm; do
    full=tcrv_emitc_ggml_repack_${sym}_q4_K_q8_K_kernel_ggml_repack_${sym}_q4_K_q8_K
    objdump -d --disassemble="$full" "$L" 2>/dev/null > "$SCR/seal_${V}_${sym}.dis"
    SP=$(grep -cE 'vs[1248]r\.v|vl[1248]re(8|16|32|64)\.v' "$SCR/seal_${V}_${sym}.dis"||true)
    VS=$(grep -cE 'vset[i]*vli' "$SCR/seal_${V}_${sym}.dis"||true)
    N16=$(grep -cE 'vsetivli[^,]*,[[:space:]]*16,' "$SCR/seal_${V}_${sym}.dis"||true)
    echo "     $sym: spill=$SP vsetvli=$VS vsetivli16=$N16[MUST0]"
  done
done

echo "=== [6] restore source byte-exact + remove .inc + pristine rebuild ==="
cp "$SCR/GEN.ORIG" "$GEN"; cp "$SCR/ARCH.ORIG" "$ARCH"
rm -f "$ARCHDIR/tcrv_emitted_gemm_q4_K.inc" "$ARCHDIR/tcrv_emitted_gevm_q4_K.inc"
GR=$(md5sum "$GEN"|awk '{print $1}'); AR=$(md5sum "$ARCH"|awk '{print $1}')
[ "$GR" = "$BASE_GEN" ] && [ "$AR" = "$BASE_ARCH" ] || { echo "*** restore mismatch"; exit 30; }
touch "$GEN" "$ARCH"; ninja -C "$B" ggml-cpu > "$SCR/mb_restore.log" 2>&1 && echo "  pristine rebuild OK" || { echo "restore build FAIL"; tail -15 "$SCR/mb_restore.log"; exit 31; }
cp -f "$LIVE" "$SCR/LIVE.pristine.gcc"
echo "  source RESTORED GEN=$GR ARCH=$AR ; live tcrv_sym=$(nm -C "$LIVE"|grep -c q4_K_q8_K_kernel||true)(expect0)"
echo "=== DONE ==="
md5sum "$SCR"/libggml-cpu.so.q4k*.gcc "$SCR"/libggml-cpu.so.q4kON.clangrepack
