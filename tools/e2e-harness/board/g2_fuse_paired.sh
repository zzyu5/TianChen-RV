#!/usr/bin/env bash
# g2_fuse_paired.sh — G2 [FUSE] rms_norm->mul paired hardware micro on ssh rvv (VLEN128).
#
# Drives g2_fuse_driver.c: cold-cache paired fused-vs-unfused wall speedup (N rounds, median+IQR,
# drift sentinel) + perf-stat DRAM-byte counters (cycles + LLC load/store-misses) + on-hardware
# ULP verify. Faithful reconstruction of the emit census (experiments/.../g2-fuse-rms-norm-mul/
# emit_census.txt). BOTH kernels are ONE source, ONE compiler+flags => preflight(0) toolchain
# symmetry is total. Main tree + build/ UNTOUCHED; nothing committed; no git stash.
#
# Usage: BOARD=rvv ROWS=8192 NEMB=4096 ROUNDS=12 CORE=8 \
#        MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs bash tools/e2e-harness/board/g2_fuse_paired.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD="${BOARD:-rvv}"
ROWS="${ROWS:-8192}"; NEMB="${NEMB:-4096}"; ROUNDS="${ROUNDS:-12}"; CORE="${CORE:-8}"
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"; CC="${CC:-clang-17}"
RDIR="/tmp/g2fuse"

echo "[g2] board=$BOARD rows=$ROWS nemb=$NEMB rounds=$ROUNDS core=$CORE march=$MARCH cc=$CC"
scp -q "$HERE/g2_fuse_driver.c" "$BOARD:$RDIR/g2_fuse_driver.c" 2>/dev/null || {
  ssh "$BOARD" "mkdir -p $RDIR"; scp -q "$HERE/g2_fuse_driver.c" "$BOARD:$RDIR/g2_fuse_driver.c"; }

ssh "$BOARD" "set -e; cd $RDIR
  echo '[preflight] freq='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq)' gov='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor)
  $CC -O3 -march=$MARCH -mabi=lp64d -o g2 g2_fuse_driver.c -lm
  echo '=== VERIFY (on-hardware ULP) ==='
  ROWS=1024 NEMB=$NEMB taskset -c $CORE ./g2 verify
  echo '=== COLD PAIRED N=$ROUNDS ==='
  ROWS=$ROWS NEMB=$NEMB taskset -c $CORE ./g2 paired $ROUNDS
  echo '=== PERF fused (5) ==='
  ROWS=$ROWS NEMB=$NEMB taskset -c $CORE perf stat -e cycles,LLC-load-misses,LLC-store-misses ./g2 fused 5 2>&1 | grep -E 'cycles|LLC|PERFMODE'
  echo '=== PERF unfused (5) ==='
  ROWS=$ROWS NEMB=$NEMB taskset -c $CORE perf stat -e cycles,LLC-load-misses,LLC-store-misses ./g2 unfused 5 2>&1 | grep -E 'cycles|LLC|PERFMODE'
"
echo "[g2] done. restore: nothing to restore (main tree + build/ untouched; scratch under $RDIR on board)."
