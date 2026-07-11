#!/usr/bin/env bash
# [G5 M1] Build the OFF (q8_0 gate off / block-dot) and ON (q8_0 gate on /
# upstream repack routing-freebie) .so variants from the SAME tcrv source tree
# (only the q8_0 gate line + engage banners differ => perfect gcc-symmetric A/B).
# Then RESTORE source to baseline and rebuild pristine (zero net source change).
# Leaves the two saved .so variants under $SCR for the swap-based phase-split.
set -uo pipefail
ATREE=/home/ubuntu/tcrv-llamacpp
BUILD=$ATREE/build-gcc15-rv64gcv
BIN=$BUILD/bin
GEN=$ATREE/ggml/src/ggml-cpu/repack.cpp
ARCH=$ATREE/ggml/src/ggml-cpu/arch/riscv/repack.cpp
SCR=/tmp/g5_q8
LIVE=$BIN/libggml-cpu.so.0.15.1
BASE_GEN=deb61a29dd079440ffdc8996b5bd2fa1
BASE_ARCH=99131cf791e30348b588423b2388e0b8
BANNER="TCRV G5-M1"
source /opt/tcrv-toolchains/env.sh
export LIBRARY_PATH=/opt/tcrv-toolchains/gcc-15.2.0/lib:${LIBRARY_PATH:-}
mkdir -p "$SCR"

echo "[g5] board=rvv openEuler VLEN128 load=$(cat /proc/loadavg)"
echo "=== [0] baseline verify + backup ==="
G0=$(md5sum "$GEN"|awk '{print $1}'); A0=$(md5sum "$ARCH"|awk '{print $1}')
echo "  GEN  now=$G0 base=$BASE_GEN"; echo "  ARCH now=$A0 base=$BASE_ARCH"
if [ "$G0" != "$BASE_GEN" ] || [ "$A0" != "$BASE_ARCH" ]; then echo "  *** NOT baseline -- ABORT"; exit 10; fi
cp "$GEN" "$SCR/repack.cpp.ORIG"; cp "$ARCH" "$SCR/arch_riscv_repack.cpp.ORIG"
echo "  backups: $SCR/{repack.cpp.ORIG,arch_riscv_repack.cpp.ORIG}"

echo "=== [1] build OFF variant (baseline source = q8_0 gate OFF / block-dot) ==="
touch "$GEN" "$ARCH"
if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/g5_build_off.log 2>&1; then echo "  OFF build OK"
else echo "  OFF build FAIL"; tail -40 /tmp/g5_build_off.log; exit 20; fi
cp -f "$LIVE" "$SCR/libggml-cpu.so.q8OFF"
OFFMD5=$(md5sum "$SCR/libggml-cpu.so.q8OFF"|awk '{print $1}')
OFFBAN=$(strings "$SCR/libggml-cpu.so.q8OFF" | grep -c "$BANNER" || true)
echo "  OFF .so md5=$OFFMD5  banner_count(expect 0)=$OFFBAN"

echo "=== [2] patch ON (flip q8_0 gate + engage banners) + build ==="
python3 "$SCR/deploy_patch_q8.py" || { echo "PATCH FAILED"; exit 11; }
touch "$GEN" "$ARCH"
if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/g5_build_on.log 2>&1; then echo "  ON build OK"
else echo "  ON build FAIL"; tail -40 /tmp/g5_build_on.log; exit 21; fi
cp -f "$LIVE" "$SCR/libggml-cpu.so.q8ON"
ONMD5=$(md5sum "$SCR/libggml-cpu.so.q8ON"|awk '{print $1}')
ONGEVM=$(strings "$SCR/libggml-cpu.so.q8ON" | grep -c "TCRV G5-M1 GEVM(q8_0_16x1 VLEN128" || true)
ONGEMM=$(strings "$SCR/libggml-cpu.so.q8ON" | grep -c "TCRV G5-M1 GEMM(q8_0_16x1 VLEN128" || true)
echo "  ON .so md5=$ONMD5  gevm_banner(expect>=1)=$ONGEVM  gemm_banner(expect>=1)=$ONGEMM"
[ "$ONMD5" != "$OFFMD5" ] && echo "  ON != OFF (variants differ, good)" || echo "  *** ON == OFF -- patch had no effect!"

echo "=== [3] restore source to baseline + rebuild pristine ==="
cp "$SCR/repack.cpp.ORIG" "$GEN"; cp "$SCR/arch_riscv_repack.cpp.ORIG" "$ARCH"
GR=$(md5sum "$GEN"|awk '{print $1}'); AR=$(md5sum "$ARCH"|awk '{print $1}')
echo "  restored source md5: GEN=$GR ARCH=$AR"
[ "$GR" = "$BASE_GEN" ] && [ "$AR" = "$BASE_ARCH" ] && echo "  SOURCE RESTORED byte-exact" || { echo "  *** restore mismatch"; exit 30; }
touch "$GEN" "$ARCH"
if cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" >/tmp/g5_build_restore.log 2>&1; then echo "  pristine rebuild OK"
else echo "  pristine rebuild FAIL"; tail -25 /tmp/g5_build_restore.log; exit 31; fi
LIVEMD5=$(md5sum "$LIVE"|awk '{print $1}')
LIVEBAN=$(strings "$LIVE" | grep -c "$BANNER" || true)
echo "  live .so md5=$LIVEMD5 (expect == OFF $OFFMD5)  banner(expect 0)=$LIVEBAN"
[ "$LIVEMD5" = "$OFFMD5" ] && echo "  LIVE == OFF-pristine (zero net change)" || echo "  NOTE: live!=OFF md5 (build nondeterminism); banner=$LIVEBAN is the real seal"
echo "=== [g5 build_variants DONE] source@baseline, variants saved:"
md5sum "$SCR/libggml-cpu.so.q8OFF" "$SCR/libggml-cpu.so.q8ON" "$LIVE"
