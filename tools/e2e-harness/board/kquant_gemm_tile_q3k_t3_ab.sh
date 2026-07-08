#!/usr/bin/env bash
# kquant_gemm_tile_q3k_t3_ab.sh -- [KQUANT-L1 / T3-tile] q3_K repack GEMM
#   UNTILED (HEAD 924dc31f front-door export, the board-harvest2 q3_K prefill 0.18x baseline) vs
#   TILED (working-tree S6-scheme: decoded signed-scale int16 stack-panel via vse16/vle16 + on-demand
#   d f16 widen deferred to the end-of-block fold; NO min-fold panel -- q3_K has no min). q3_K is the
#   no-min analogue of the q4_K S6 register-cliff step: stage the idle-across-the-hot-dot decode strips
#   to STACK PANELS so the hot main-term dot's peak live-vreg fits <=32 and allocator live-value spills
#   go to ~0 -- byte-exact-preserving (single end-of-block f32 convert+fold kept in the SAME order).
#
#   UNTILED = untiled_q3K_gemm.c  (md5 0f14791e; HEAD 924dc31f export == kquant-l1-q6q2q3-repack cell)
#   TILED   = tiled_q3K_gemm.c    (md5 28cbeebd; working-tree emitter delta, regen via read-only tcrv-opt)
#   GEVM (both, for the verify gate; direct-connect, not tiled): untiled_q3K_gevm.c / tiled_q3K_gevm.c
#
#   Measures: (1) objdump vsetvli + spill(vs{1,2,4,8}r.v)/reload(vl{1,2,4,8}r.v) + maxVreg + vwmacc
#   UNTILED->TILED at clang-17 -O2 (measured) AND -O3 (diagnosis) -- does spill reach ~0, peak <=32,
#   vwmacc multiset preserved? (2) silicon numerical verify (kquant_repack_verify_q3K) linked with
#   UNTILED then TILED objs: INT byte-exact-integer (mismatch=0) + NORM bounded-ULP for both -> both
#   compute the oracle => the timed TILED is byte-exact. (3) IDENTITY-DUMP untiled-vs-tiled full fp32
#   output cmp on the driver's small-finite fp16-scale fold path (byte-identical?). (4) A/B tiled/untiled
#   paired cold N rounds (tiling-isolated incremental speedup). (5) vs-opponent parity ours/opp for
#   UNTILED and TILED (untiled was ~0.18x; does tiling reach parity/win?).
#
#   Both q3_K kernels compiled clang-17 -O2 into SEPARATE binaries, each linked against the board's OWN
#   libggml-cpu.so (opponent = real dispatched ggml_vec_dot_q3_K_q8_K block-dot). Cold N rounds: each
#   round runs untiled then tiled as FRESH processes. Main tree + build/ UNTOUCHED; nothing committed;
#   no git stash (untiled is the cached HEAD export). [NG-4]: L1 kernel datapoint, NOT a beat.
#
# Usage: BOARD=rvv CORE=8 K=2048 NR=64 NC=512 ITERS=20 N=12 bash tools/e2e-harness/board/kquant_gemm_tile_q3k_t3_ab.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD="${BOARD:-rvv}"; CORE="${CORE:-8}"; K="${K:-2048}"; NR="${NR:-64}"; NC="${NC:-512}"
ITERS="${ITERS:-20}"; N="${N:-12}"
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"; CC="${CC:-clang-17}"
GGML_BIN="${GGML_BIN:-/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin}"
SRC="${SRC:-/tmp/q3k_tile_t3}"    # host: untiled_q3K_gemm.c tiled_q3K_gemm.c untiled_q3K_gevm.c tiled_q3K_gevm.c
RDIR="/tmp/kquant_tile_q3k_t3_ab"

echo "[tile-q3k-t3-ab] board=$BOARD core=$CORE K=$K nr=$NR nc=$NC iters=$ITERS N=$N march=$MARCH cc=$CC"
ssh "$BOARD" "mkdir -p $RDIR"
scp -q "$SRC/untiled_q3K_gemm.c" "$SRC/tiled_q3K_gemm.c" "$SRC/untiled_q3K_gevm.c" "$SRC/tiled_q3K_gevm.c" \
       "$HERE/kquant_gemm_paired_q3k_driver.c" "$HERE/kquant_repack_verify_q3K.c" "$BOARD:$RDIR/"

ssh "$BOARD" "set -e; cd $RDIR
  echo '[preflight] loadavg='\$(cat /proc/loadavg)' core=$CORE freq='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null||echo NA)' gov='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null||echo NA)
  ls $GGML_BIN/libggml-cpu.so >/dev/null || { echo 'GGML_BIN missing libggml-cpu.so'; exit 40; }

  echo '[compile] untiled/tiled q3_K GEMM at -O2 (measured) and -O3 (diagnosis); GEVM (both); driver; verify'
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ untiled_q3K_gemm.c -c -o u_gemm_O2.o
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ tiled_q3K_gemm.c   -c -o t_gemm_O2.o
  $CC -O3 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ untiled_q3K_gemm.c -c -o u_gemm_O3.o
  $CC -O3 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ tiled_q3K_gemm.c   -c -o t_gemm_O3.o
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ untiled_q3K_gevm.c -c -o u_gevm_O2.o
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ tiled_q3K_gevm.c   -c -o t_gevm_O2.o
  $CC -O2 -march=$MARCH -mabi=lp64d -x c   kquant_gemm_paired_q3k_driver.c -c -o drv.o
  $CC -O2 -march=$MARCH -mabi=lp64d -x c++ kquant_repack_verify_q3K.c      -c -o vrf.o
  $CC drv.o u_gemm_O2.o -o q3k_untiled -L$GGML_BIN -Wl,-rpath,$GGML_BIN -lggml-cpu -lggml-base -lggml -lm
  $CC drv.o t_gemm_O2.o -o q3k_tiled   -L$GGML_BIN -Wl,-rpath,$GGML_BIN -lggml-cpu -lggml-base -lggml -lm
  $CC vrf.o u_gemm_O2.o u_gevm_O2.o -o verify_untiled -lstdc++ -lm
  $CC vrf.o t_gemm_O2.o t_gevm_O2.o -o verify_tiled   -lstdc++ -lm

  echo '=== [1] OBJDUMP seal: vsetvli + spill/reload + maxVreg + vwmacc  UNTILED->TILED (spill ~0? peak <=32?) ==='
  seal(){  # \$1=objfile \$2=label
    local DIS; DIS=\$(objdump -d \$1 2>/dev/null)
    local vs=\$(  printf '%s' \"\$DIS\" | grep -cE 'vsetvli|vsetivli' || true )
    local sp=\$(  printf '%s' \"\$DIS\" | grep -cE 'vs[1248]r\\.v'    || true )
    local rl=\$(  printf '%s' \"\$DIS\" | grep -cE 'vl[1248]r\\.v'    || true )
    local wm=\$(  printf '%s' \"\$DIS\" | grep -cE 'vwmacc'           || true )
    local mv=\$(  printf '%s' \"\$DIS\" | grep -oE '[ ,(]v[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1 )
    local tx=\$(  objdump -h \$1 2>/dev/null | awk '/\\.text/{print strtonum(\"0x\"\$3)}' )
    printf '  OBJD %-12s vsetvli=%-4s spill=%-4s reload=%-4s vwmacc=%-5s maxVreg=v%-3s textB=%s\\n' \"\$2\" \"\$vs\" \"\$sp\" \"\$rl\" \"\$wm\" \"\$mv\" \"\$tx\"
  }
  seal u_gemm_O2.o 'UNTILED -O2'; seal t_gemm_O2.o 'TILED   -O2'
  seal u_gemm_O3.o 'UNTILED -O3'; seal t_gemm_O3.o 'TILED   -O3'
  echo '--- hot-loop spill classification (TILED -O2): vs*r.v addresses vs first hot-dot label ---'
  objdump -d t_gemm_O2.o 2>/dev/null | grep -nE 'vs[1248]r\\.v' | head -20 || true

  echo '=== [2] SILICON verify (INT byte-exact-integer + NORM bounded-ULP) UNTILED then TILED ==='
  echo '--- verify UNTILED ---'; taskset -c $CORE ./verify_untiled 0xC0FFEE | tail -4
  echo '--- verify TILED   ---'; taskset -c $CORE ./verify_tiled   0xC0FFEE | tail -4

  echo '=== [3] IDENTITY-DUMP byte-exact cmp UNTILED vs TILED (small-finite fp16-scale fold path) ==='
  Q3K_DUMP=$RDIR/o_untiled.bin taskset -c $CORE ./q3k_untiled q3_K $K $NR $NC 10 0xC0FFEE
  Q3K_DUMP=$RDIR/o_tiled.bin   taskset -c $CORE ./q3k_tiled   q3_K $K $NR $NC 10 0xC0FFEE
  if cmp -s o_untiled.bin o_tiled.bin; then echo '  IDENTITY: o_untiled.bin == o_tiled.bin  (BYTE-EXACT, 0 mismatch)'; \
    else echo '  IDENTITY: *** MISMATCH *** untiled vs tiled full fp32 output'; cmp o_untiled.bin o_tiled.bin | head; fi

  echo '=== [4/5] A/B(tiled vs untiled) + vs-opponent (cold N=$N rounds; each round untiled then tiled fresh procs) ==='
  echo '    KQGEMM line: OURS best_gmacs / OPP best_gmacs / ratio_best (ours/opp)'
  echo '[loadavg-mid] '\$(cat /proc/loadavg)
  for i in \$(seq 1 $N); do
    r_u=\$( LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./q3k_untiled q3_K $K $NR $NC $ITERS 0xC0FFEE )
    echo \"ROUND \$i untiled \$r_u\"
    r_t=\$( LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./q3k_tiled   q3_K $K $NR $NC $ITERS 0xC0FFEE )
    echo \"ROUND \$i tiled   \$r_t\"
  done
  echo '[loadavg-end] '\$(cat /proc/loadavg)
"
echo "[tile-q3k-t3-ab] done. restore: main tree + build/ UNTOUCHED (no rebuild); board scratch ephemeral under $RDIR; governor left as-found."
