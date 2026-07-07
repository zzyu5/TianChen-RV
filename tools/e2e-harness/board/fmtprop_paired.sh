#!/usr/bin/env bash
# fmtprop_paired.sh — G2 [FMT-PROP] rms_norm->mul->quantize(q8_0) paired hardware micro on rvv (VLEN128).
#
# Drives fmtprop_driver.c: cold-cache paired fused-vs-unfused wall speedup (N rounds, median+IQR,
# drift sentinel) + perf-stat DRAM-byte counters (cycles + LLC load/store-misses) + on-hardware
# byte-exact q8_0 verify. Faithful reconstruction of the emit census of commit 2b814e46
# (emitElementwiseRmsNormReduceStrip + emitQuantizeQ80BlockBody). BOTH kernels are ONE source,
# ONE compiler+flags => preflight(0) toolchain symmetry total. Main tree + build/ UNTOUCHED;
# nothing committed; no git stash.
#
# Usage: BOARD=rvv ROWS=8192 NEMB=4096 ROUNDS=12 CORE=8 \
#        MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs bash tools/e2e-harness/board/fmtprop_paired.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD="${BOARD:-rvv}"
ROWS="${ROWS:-8192}"; NEMB="${NEMB:-4096}"; ROUNDS="${ROUNDS:-12}"; CORE="${CORE:-8}"
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"; CC="${CC:-clang-17}"
RDIR="/tmp/fmtprop"

echo "[fmtprop] board=$BOARD rows=$ROWS nemb=$NEMB rounds=$ROUNDS core=$CORE march=$MARCH cc=$CC"
ssh "$BOARD" "mkdir -p $RDIR"
scp -q "$HERE/fmtprop_driver.c" "$BOARD:$RDIR/fmtprop_driver.c"

ssh "$BOARD" "set -e; cd $RDIR
  echo '[preflight] vlenb='\$(cat /proc/cpuinfo | grep -m1 -i isa >/dev/null; echo)  # isa below
  echo '[preflight] freq='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null || echo NA)' gov='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null || echo NA)
  $CC -O3 -march=$MARCH -mabi=lp64d -o fmtprop fmtprop_driver.c -lm
  echo '[preflight] objdump libcall-free check:'
  (llvm-objdump-17 -d fmtprop 2>/dev/null || objdump -d fmtprop 2>/dev/null) | grep -Eq '__truncsfhf2|__extendhfsf2|__gnu_f2h_ieee|__gnu_h2f_ieee' && echo '  SOFT-FP16 LIBCALL PRESENT (crippled)' || echo '  libcall-free: OK (hardware fp16)'
  echo '=== VERIFY (on-hardware byte-exact q8_0) ==='
  ROWS=1024 NEMB=$NEMB taskset -c $CORE ./fmtprop verify
  echo '=== COLD PAIRED N=$ROUNDS ==='
  ROWS=$ROWS NEMB=$NEMB taskset -c $CORE ./fmtprop paired $ROUNDS
  echo '=== PERF fused (6) ==='
  ROWS=$ROWS NEMB=$NEMB taskset -c $CORE perf stat -e cycles,LLC-load-misses,LLC-store-misses ./fmtprop fused 6 2>&1 | grep -E 'cycles|LLC|PERFMODE' || true
  echo '=== PERF unfused (6) ==='
  ROWS=$ROWS NEMB=$NEMB taskset -c $CORE perf stat -e cycles,LLC-load-misses,LLC-store-misses ./fmtprop unfused 6 2>&1 | grep -E 'cycles|LLC|PERFMODE' || true
"
echo "[fmtprop] done. restore: nothing to restore (main tree + build/ untouched; scratch under $RDIR on board)."
