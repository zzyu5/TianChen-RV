#!/usr/bin/env bash
# [decisive-kquant-gcc-vs-vlen] Build ggml-cpu with CLANG-17 in a dedicated tree.
# Produces q4kOFF (baseline block-dot) + q4kON (our EMITTED vl=8 q4_K repack intercept),
# both clang-17 => clang-symmetric account. Same A-tree source as gcc build; source
# patched then RESTORED byte-exact. Only the ggml-cpu target is built (BUILD_SHARED_LIBS
# => swappable libggml-cpu.so). NO git.
set -uo pipefail
ATREE=/home/ubuntu/tcrv-llamacpp
CBUILD=$ATREE/build-clang17-dkgv
GEN=$ATREE/ggml/src/ggml-cpu/repack.cpp
ARCH=$ATREE/ggml/src/ggml-cpu/arch/riscv/repack.cpp
ARCHDIR=$ATREE/ggml/src/ggml-cpu/arch/riscv
SCR=/tmp/dkgv
BASE_GEN=deb61a29dd079440ffdc8996b5bd2fa1
BASE_ARCH=99131cf791e30348b588423b2388e0b8
# clang-18 (TCRV LLVM): has __riscv_vcreate_v_* intrinsics (clang-17 lacks them);
# needs explicit -B/-L to the openEuler gcc-12 runtime (triple mismatch w/ ld.lld).
CC=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang; CXX=/opt/tcrv-toolchains/llvm-18.1.8/bin/clang++
GCCRT=/usr/lib/gcc/riscv64-openEuler-linux/12
ISYS="-isystem /usr/include/c++/12 -isystem /usr/include/c++/12/riscv64-openEuler-linux -isystem /usr/include/c++/12/backward"
MARCH="rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zicbop_zihintpause"
FLAGS="-march=$MARCH -mabi=lp64d"
CXXFLAGS="-march=$MARCH -mabi=lp64d $ISYS"
LINKFLAGS="-B$GCCRT -L$GCCRT -L/usr/lib64"
LIB=libggml-cpu.so.0.15.1
CML=$ATREE/ggml/src/ggml-cpu/CMakeLists.txt

echo "[dkgv] board=rvv VLEN128 load=$(cat /proc/loadavg) clang=$($CC --version|head -1)"
echo "=== [0] baseline verify + backup source ==="
G0=$(md5sum "$GEN"|awk '{print $1}'); A0=$(md5sum "$ARCH"|awk '{print $1}')
echo "  GEN now=$G0 base=$BASE_GEN ; ARCH now=$A0 base=$BASE_ARCH"
BASE_CML=4426f54358c6958f4dc317eb9a3bb1bf
C0=$(md5sum "$CML"|awk '{print $1}')
if [ "$G0" != "$BASE_GEN" ] || [ "$A0" != "$BASE_ARCH" ]; then echo "*** src NOT baseline -- ABORT"; exit 10; fi
if [ "$C0" != "$BASE_CML" ]; then echo "*** CMakeLists NOT baseline ($C0 != $BASE_CML) -- restore first, ABORT"; exit 10; fi
cp "$GEN" "$SCR/GEN.ORIG"; cp "$ARCH" "$SCR/ARCH.ORIG"; cp "$CML" "$SCR/CML.ORIG"
CML_MD5=$(md5sum "$CML"|awk '{print $1}'); echo "  CMakeLists md5=$CML_MD5 (backed up)"

echo "=== [0b] patch ggml-cpu L1-SEAL MARCH_STR clang-conditional (drop zfa/zvfhmin/zicond/zawrs experimental) ==="
# Replace the unconditional L1-SEAL override with a compiler-conditional one.
python3 - "$CML" <<PYEOF
import sys,re
p=sys.argv[1]; s=open(p).read()
old='            set(MARCH_STR "rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause")  # L1-SEAL full-capability override'
new='''            if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
                set(MARCH_STR "rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zicbop_zihintpause")  # DKGV clang-compat (zfa/zvfhmin/zicond/zawrs experimental in clang-17)
            else()
                set(MARCH_STR "rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause")  # L1-SEAL full-capability override
            endif()'''
assert s.count(old)==1, f"anchor count={s.count(old)}"
open(p,"w").write(s.replace(old,new)); print("  CMakeLists MARCH_STR patched clang-conditional")
PYEOF
[ $? -eq 0 ] || { echo "CML patch FAIL"; exit 13; }

echo "=== [1] configure clang tree (fresh) ==="
rm -rf "$CBUILD"
if true; then
  cmake -S "$ATREE" -B "$CBUILD" -G Ninja \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_C_FLAGS="$FLAGS" -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="$LINKFLAGS" -DCMAKE_SHARED_LINKER_FLAGS="$LINKFLAGS" \
    -DGGML_NATIVE=OFF -DGGML_CPU_REPACK=ON -DGGML_OPENMP=ON \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_CURL=OFF \
    > "$SCR/cfg.log" 2>&1 || { echo "CONFIGURE FAIL"; tail -30 "$SCR/cfg.log"; exit 12; }
  echo "  configured OK"
else echo "  cache exists, skip configure"; fi

BIN=$CBUILD/bin
echo "=== [2] build OFF (baseline source = q4_K gate OFF / block-dot) ==="
touch "$GEN" "$ARCH"
cmake --build "$CBUILD" --target ggml-cpu -j"$(nproc)" > "$SCR/build_off.log" 2>&1 || { echo "OFF BUILD FAIL"; tail -40 "$SCR/build_off.log"; exit 20; }
cp -f "$BIN/$LIB" "$SCR/libggml-cpu.so.q4kOFF.clang17"
OFFMD5=$(md5sum "$SCR/libggml-cpu.so.q4kOFF.clang17"|awk '{print $1}')
OFFSYM=$(nm -C "$SCR/libggml-cpu.so.q4kOFF.clang17" 2>/dev/null | grep -c "tcrv_emitc_ggml_repack_gem._q4_K_q8_K_kernel" || true)
echo "  OFF md5=$OFFMD5 q4_K_tcrv_sym(expect 0)=$OFFSYM"

echo "=== [3] patch ON (emitted vl=8 intercept) + build ==="
cp -f "$SCR/tcrv_emitted_gemm_q4_K.inc" "$ARCHDIR/tcrv_emitted_gemm_q4_K.inc"
cp -f "$SCR/tcrv_emitted_gevm_q4_K.inc" "$ARCHDIR/tcrv_emitted_gevm_q4_K.inc"
python3 "$SCR/deploy_patch_q4k_emitted.py" || { echo "PATCH FAIL"; exit 11; }
touch "$GEN" "$ARCH"
cmake --build "$CBUILD" --target ggml-cpu -j"$(nproc)" > "$SCR/build_on.log" 2>&1 || { echo "ON BUILD FAIL"; tail -60 "$SCR/build_on.log"; exit 21; }
cp -f "$BIN/$LIB" "$SCR/libggml-cpu.so.q4kON.clang17"
ONMD5=$(md5sum "$SCR/libggml-cpu.so.q4kON.clang17"|awk '{print $1}')
echo "  ON md5=$ONMD5"

echo "=== [4] SEAL ON: nm symbols + banner + objdump vl=8 + spill ==="
nm -C "$SCR/libggml-cpu.so.q4kON.clang17" 2>/dev/null | grep "tcrv_emitc_ggml_repack_gem._q4_K_q8_K_kernel" | sed 's/^/    /'
echo "  banner gevm=$(strings "$SCR/libggml-cpu.so.q4kON.clang17"|grep -c 'TCRV G5-M2 EMITTED GEVM(q4_K_16x1 VLEN128')  gemm=$(strings "$SCR/libggml-cpu.so.q4kON.clang17"|grep -c 'TCRV G5-M2 EMITTED GEMM(q4_K_16x1 VLEN128')"
[ "$ONMD5" != "$OFFMD5" ] && echo "  ON != OFF (differ, good)" || echo "  *** ON == OFF!"
for sym in gemv gemm; do
  full=tcrv_emitc_ggml_repack_${sym}_q4_K_q8_K_kernel_ggml_repack_${sym}_q4_K_q8_K
  objdump -d --disassemble="$full" "$SCR/libggml-cpu.so.q4kON.clang17" 2>/dev/null > "$SCR/seal_$sym.dis"
  N8=$(grep -cE 'vsetivli[^,]*,[[:space:]]*8,' "$SCR/seal_$sym.dis" || true)
  N16=$(grep -cE 'vsetivli[^,]*,[[:space:]]*16,' "$SCR/seal_$sym.dis" || true)
  SP=$(grep -cE 'vs[1248]r\.v|vl[1248]re(8|16|32|64)\.v' "$SCR/seal_$sym.dis" || true)
  VS=$(grep -cE 'vset[i]*vli' "$SCR/seal_$sym.dis" || true)
  echo "    $sym: vsetivli(imm=8)=$N8 vsetivli(imm=16)=$N16[MUST0] spill=$SP vsetvli_total=$VS"
done

echo "=== [5] restore source byte-exact + remove .inc + restore CMakeLists ==="
cp "$SCR/GEN.ORIG" "$GEN"; cp "$SCR/ARCH.ORIG" "$ARCH"; cp "$SCR/CML.ORIG" "$CML"
rm -f "$ARCHDIR/tcrv_emitted_gemm_q4_K.inc" "$ARCHDIR/tcrv_emitted_gevm_q4_K.inc"
GR=$(md5sum "$GEN"|awk '{print $1}'); AR=$(md5sum "$ARCH"|awk '{print $1}'); CR=$(md5sum "$CML"|awk '{print $1}')
[ "$GR" = "$BASE_GEN" ] && [ "$AR" = "$BASE_ARCH" ] && [ "$CR" = "$CML_MD5" ] && echo "  SOURCE RESTORED byte-exact GEN=$GR ARCH=$AR CML=$CR" || { echo "*** restore mismatch GEN=$GR ARCH=$AR CML=$CR(base $CML_MD5)"; exit 30; }
echo "=== DONE variants ==="
md5sum "$SCR/libggml-cpu.so.q4kOFF.clang17" "$SCR/libggml-cpu.so.q4kON.clang17"
