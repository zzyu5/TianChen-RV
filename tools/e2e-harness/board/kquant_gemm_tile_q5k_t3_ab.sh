#!/usr/bin/env bash
# kquant_gemm_tile_q5k_t3_ab.sh -- [KQUANT-L1 / T3-tile] q5_K repack GEMM (the LAST K-quant)
#   UNTILED (pre-tiling front-door export k_gemm_q5K.cpp, oracle-GREEN @construction) vs
#   TILED (working-tree S6-scheme: q4_K S6-tiled GEMM body + the qh 5th-bit inject -- decoded
#   dual d/dmin + 6-bit scale/min int16 stack-panels via vse16/vle16 + inline-min-fold i32 bsums
#   stack-panel via vse32/vle32 + on-demand d/dmin f16 widen deferred to the end-of-block fold).
#   q5_K SHARES the q4_K min fold (kquant_dmin_bsums_min, dual d/dmin + bsums-min), so the FAMILY
#   S6 register-cliff lever should transfer WHOLE like q2_K -- UNLESS the qh 5th-bit plane makes
#   the peak pressure weight-bound like q6_K/q3_K. ★[XFER-1] verify #2: does q5_K reach the <=32
#   register cliff (v30, min-fold dominates, HOLDS) or stay v31 (qh plane weight-bound, NULL)?
#   Pre-registered gate: spill->~0 ^ MAC/cycle up ^ wall T-N ==> tiling HOLDS.
#
#   UNTILED = untiled_q5K_gemm.c  (k_gemm_q5K.cpp; pre-S6 front-door export, 24 panel markers,
#             vwmacc 4480; oracle-GREEN WORST_NORM 8.0e-07 on-board @construction)
#   TILED   = tiled_q5K_gemm.c    (working-tree emitter delta emitRepackKQuantGemmBodyQ5K; 440
#             panel markers, vwmacc 4480 UNCHANGED; regen via read-only build/bin/tcrv-opt on the
#             campaign-canonical fixture rvv-to-emitc-repack-gemm-q5-K-q8-K.mlir)
#   ORACLE  = oracle_q5K.cpp      (INDEPENDENT scalar q5_K dequant-matmul reference, GEMM-only;
#             qh/NOMIN/PERM/ROWROT control-flips prove sensitivity to the q5_K axis)
#
#   Measures: (1) objdump vsetvli + spill(vs{1,2,4,8}r.v)/reload(vl{1,2,4,8}r.v) + maxVreg + vwmacc
#   UNTILED->TILED at clang-17 -O2 (measured) AND -O3 (diagnosis) -- does spill reach ~0, peak <=32,
#   vwmacc multiset preserved? (2) silicon oracle verify linked with UNTILED then TILED objs: both
#   NORM<1e-4 => both compute the q5_K oracle => the timed TILED is correct; UNTILED --controls
#   prints the qh/min/interleave flip margins. (3) IDENTITY-DUMP untiled-vs-tiled full fp32 output
#   cmp on the driver's small-finite dual-fp16-scale/min fold path (byte-identical?). (4) A/B
#   tiled/untiled paired cold N rounds (tiling-isolated incremental speedup). (5) vs-opponent parity
#   ours/opp for UNTILED and TILED (untiled was ~1.5x vs UNTUNED; does tiling push it further?).
#
#   Both q5_K kernels compiled clang-17 -O2 into SEPARATE binaries, each linked against the board's
#   OWN libggml-cpu.so (opponent = real dispatched ggml_vec_dot_q5_K_q8_K block-dot). Cold N rounds:
#   each round runs untiled then tiled as FRESH processes. Main tree + build/ UNTOUCHED; nothing
#   committed; no git stash (untiled is the cached pre-S6 export). [NG-4]: L1 kernel datapoint.
#
# Usage: BOARD=rvv CORE=8 K=2048 NR=64 NC=512 ITERS=20 N=12 bash tools/e2e-harness/board/kquant_gemm_tile_q5k_t3_ab.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD="${BOARD:-rvv}"; CORE="${CORE:-8}"; K="${K:-2048}"; NR="${NR:-64}"; NC="${NC:-512}"
ITERS="${ITERS:-20}"; N="${N:-12}"
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"; CC="${CC:-clang-17}"
GGML_BIN="${GGML_BIN:-/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin}"
SRC="${SRC:-/tmp/q5k_tile_t3}"    # host: untiled_q5K_gemm.c tiled_q5K_gemm.c oracle_q5K.cpp
RDIR="/tmp/kquant_tile_q5k_t3_ab"

echo "[tile-q5k-t3-ab] board=$BOARD core=$CORE K=$K nr=$NR nc=$NC iters=$ITERS N=$N march=$MARCH cc=$CC"
ssh "$BOARD" "mkdir -p $RDIR"
scp -q "$SRC/untiled_q5K_gemm.c" "$SRC/tiled_q5K_gemm.c" "$SRC/oracle_q5K.cpp" \
       "$HERE/kquant_gemm_paired_q5k_driver.c" "$BOARD:$RDIR/"

ssh "$BOARD" "set -e; cd $RDIR
  echo '[preflight] loadavg='\$(cat /proc/loadavg)' core=$CORE freq='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null||echo NA)' gov='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null||echo NA)
  ls $GGML_BIN/libggml-cpu.so >/dev/null || { echo 'GGML_BIN missing libggml-cpu.so'; exit 40; }

  echo '[compile] untiled/tiled q5_K GEMM at -O2 (measured) and -O3 (diagnosis); oracle; driver'
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ untiled_q5K_gemm.c -c -o u_gemm_O2.o
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ tiled_q5K_gemm.c   -c -o t_gemm_O2.o
  $CC -O3 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ untiled_q5K_gemm.c -c -o u_gemm_O3.o
  $CC -O3 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ tiled_q5K_gemm.c   -c -o t_gemm_O3.o
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ oracle_q5K.cpp     -c -o orc.o
  $CC -O2 -march=$MARCH -mabi=lp64d -x c   kquant_gemm_paired_q5k_driver.c -c -o drv.o
  $CC drv.o u_gemm_O2.o -o q5k_untiled -L$GGML_BIN -Wl,-rpath,$GGML_BIN -lggml-cpu -lggml-base -lggml -lm
  $CC drv.o t_gemm_O2.o -o q5k_tiled   -L$GGML_BIN -Wl,-rpath,$GGML_BIN -lggml-cpu -lggml-base -lggml -lm
  $CC orc.o u_gemm_O2.o -o verify_untiled -lstdc++ -lm
  $CC orc.o t_gemm_O2.o -o verify_tiled   -lstdc++ -lm

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
  echo '--- hot-loop spill classification (TILED -O2): vs*r.v addresses (first 20) ---'
  objdump -d t_gemm_O2.o 2>/dev/null | grep -nE 'vs[1248]r\\.v' | head -20 || true

  echo '=== [2] SILICON oracle verify (independent q5_K dequant-matmul, NORM<1e-4) UNTILED then TILED ==='
  echo '--- verify UNTILED (+ --controls: qh/min/interleave flip margins) ---'
  taskset -c $CORE ./verify_untiled --controls 2>&1 | grep -E 'WORST_NORM|VERDICT|control|BUG' | head -12
  echo '--- verify TILED ---'
  taskset -c $CORE ./verify_tiled   2>&1 | grep -E 'WORST_NORM|VERDICT|BUG' | head -4

  echo '=== [3] IDENTITY-DUMP byte-exact cmp UNTILED vs TILED (small-finite dual-fp16-scale/min fold path) ==='
  Q5K_DUMP=$RDIR/o_untiled.bin taskset -c $CORE ./q5k_untiled q5_K $K $NR $NC 10 0xC0FFEE
  Q5K_DUMP=$RDIR/o_tiled.bin   taskset -c $CORE ./q5k_tiled   q5_K $K $NR $NC 10 0xC0FFEE
  if cmp -s o_untiled.bin o_tiled.bin; then echo '  IDENTITY: o_untiled.bin == o_tiled.bin  (BYTE-EXACT, 0 mismatch)'; \
    else echo '  IDENTITY: *** MISMATCH *** untiled vs tiled full fp32 output'; cmp o_untiled.bin o_tiled.bin | head; fi

  echo '=== [4/5] A/B(tiled vs untiled) + vs-opponent (cold N=$N rounds; each round untiled then tiled fresh procs) ==='
  echo '    KQGEMM line: OURS best_gmacs / OPP best_gmacs / ratio_best (ours/opp)'
  echo '[loadavg-mid] '\$(cat /proc/loadavg)
  for i in \$(seq 1 $N); do
    r_u=\$( LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./q5k_untiled q5_K $K $NR $NC $ITERS 0xC0FFEE )
    echo \"ROUND \$i untiled \$r_u\"
    r_t=\$( LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./q5k_tiled   q5_K $K $NR $NC $ITERS 0xC0FFEE )
    echo \"ROUND \$i tiled   \$r_t\"
  done
  echo '[loadavg-end] '\$(cat /proc/loadavg)
"
echo "[tile-q5k-t3-ab] done. restore: main tree + build/ UNTOUCHED (no rebuild); board scratch ephemeral under $RDIR; governor left as-found."
