#!/usr/bin/env bash
# kquant_gemm_reroll_ab.sh — [KQUANT-L1 / RE-ROLL, 曳光弹 #2] q4_K repack GEMM
#   PRE (full-unroll golden) vs POST (k/ii runtime for-loop RE-ROLL) A/B on rvv/VLEN128,
#   plus vs-opponent block-dot parity. The pipeline reorder (#1, byte-exact) did NOT convert
#   the full-unroll register-pressure opening into throughput; this measures whether the
#   NON-byte-exact structural RE-ROLL (turn the fully-unrolled superblock's k-chunk + ii-nibble
#   dims into runtime emitc.for loops so clang can hoist the loop-invariant vsetvli / reuse
#   accumulator registers) does.
#
#   PRE  = /tmp/q4k_reroll/pre_q4K.c   (== cached golden baseline md5 b0b5beac; oracle-GREEN @039133ea)
#   POST = /tmp/q4k_reroll/post_q4K.c  (re-rolled; byte-exact-identical per verify_q4k_identity gate)
#
#   Measures: (1) objdump vsetvli + spill(vs{1,2,4,8}r.v)/reload(vl{1,2,4,8}r.v) PRE->POST at the
#   MEASURED board form (clang-17 -O2) AND the diagnosis form (clang-17 -O3) — does it DROP or RISE?
#   (2) byte-exact identity gate (int + norm) PRE vs POST via `cmp` (correctness of the timed binary).
#   (3) A/B ours_post/ours_pre paired cold N rounds (reroll isolated speedup).
#   (4) vs-opponent parity ours/opp for PRE and POST (does the precedent 0.962x move / flip to win?).
#
#   Both q4_K kernels compiled clang-17 -O2 (SAME flags as kquant_gemm_paired.sh) into SEPARATE
#   binaries, each linked against the board's OWN libggml-cpu.so (opponent = real dispatched
#   block-dot). Cold N rounds: each round runs pre then post as FRESH processes. Main tree + build/
#   UNTOUCHED; nothing committed; no git stash. [NG-4]: L1 datapoint, NOT a beat (eight gates unwalked).
#
# Usage: BOARD=rvv CORE=8 K=2048 NR=64 NC=512 ITERS=20 N=12 bash tools/e2e-harness/board/kquant_gemm_reroll_ab.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD="${BOARD:-rvv}"; CORE="${CORE:-8}"; K="${K:-2048}"; NR="${NR:-64}"; NC="${NC:-512}"
ITERS="${ITERS:-20}"; N="${N:-12}"
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"; CC="${CC:-clang-17}"
GGML_BIN="${GGML_BIN:-/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin}"
SRC="${SRC:-/tmp/q4k_reroll}"     # host dir: pre_q4K.c(golden) post_q4K.c(reroll) gemm_q5_K_q8_K.kernel.c verify_q4k_identity.c
RDIR="/tmp/kquant_reroll_ab"

echo "[reroll-ab] board=$BOARD core=$CORE K=$K nr=$NR nc=$NC iters=$ITERS N=$N march=$MARCH cc=$CC"
ssh "$BOARD" "mkdir -p $RDIR"
scp -q "$SRC/pre_q4K.c" "$SRC/post_q4K.c" "$SRC/gemm_q5_K_q8_K.kernel.c" "$SRC/verify_q4k_identity.c" \
       "$HERE/kquant_gemm_paired_driver.c" "$BOARD:$RDIR/"

ssh "$BOARD" "set -e; cd $RDIR
  echo '[preflight] core=$CORE freq='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null||echo NA)' gov='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null||echo NA)
  ls $GGML_BIN/libggml-cpu.so >/dev/null || { echo 'GGML_BIN missing libggml-cpu.so'; exit 40; }

  echo '[compile] PRE/POST q4_K at -O2 (measured) and -O3 (diagnosis); shared q5_K + paired driver (-O2)'
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ pre_q4K.c  -c -o pre_O2.o
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ post_q4K.c -c -o post_O2.o
  $CC -O3 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ pre_q4K.c  -c -o pre_O3.o
  $CC -O3 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ post_q4K.c -c -o post_O3.o
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ gemm_q5_K_q8_K.kernel.c -c -o kq5k.o
  $CC -O2 -march=$MARCH -mabi=lp64d -x c kquant_gemm_paired_driver.c -c -o kqdrv.o
  $CC kqdrv.o pre_O2.o  kq5k.o -o kqgemm_pre  -L$GGML_BIN -Wl,-rpath,$GGML_BIN -lggml-cpu -lggml-base -lggml -lm
  $CC kqdrv.o post_O2.o kq5k.o -o kqgemm_post -L$GGML_BIN -Wl,-rpath,$GGML_BIN -lggml-cpu -lggml-base -lggml -lm

  echo '=== [1] OBJDUMP seal: vsetvli + spill/reload PRE->POST (does it DROP?) ==='
  seal(){  # \$1=objfile \$2=label
    local DIS; DIS=\$(objdump -d \$1 2>/dev/null)
    local vs=\$(  printf '%s' \"\$DIS\" | grep -cE 'vsetvli|vsetivli' || true )
    local sp=\$(  printf '%s' \"\$DIS\" | grep -cE 'vs[1248]r\\.v'    || true )
    local rl=\$(  printf '%s' \"\$DIS\" | grep -cE 'vl[1248]r\\.v'    || true )
    local wm=\$(  printf '%s' \"\$DIS\" | grep -cE 'vwmacc'           || true )
    local tx=\$(  objdump -h \$1 2>/dev/null | awk '/\\.text/{print strtonum(\"0x\"\$3)}' )
    printf '  OBJD %-10s vsetvli=%-4s spill=%-4s reload=%-4s vwmacc=%-5s textB=%s\\n' \"\$2\" \"\$vs\" \"\$sp\" \"\$rl\" \"\$wm\" \"\$tx\"
  }
  seal pre_O2.o  'PRE  -O2'; seal post_O2.o 'POST -O2'
  seal pre_O3.o  'PRE  -O3'; seal post_O3.o 'POST -O3'

  echo '=== [2] BYTE-EXACT identity gate PRE vs POST (int + norm) ==='
  $CC -O2 -march=$MARCH -mabi=lp64d -x c verify_q4k_identity.c -c -o vdrv.o
  $CC vdrv.o pre_O2.o  -o verify_pre  -lm
  $CC vdrv.o post_O2.o -o verify_post -lm
  gate_ok=1
  for mode in int norm; do
    taskset -c $CORE ./verify_pre  \$mode $K $NR $NC 0xC0FFEE opre_\$mode  >/dev/null
    taskset -c $CORE ./verify_post \$mode $K $NR $NC 0xC0FFEE opost_\$mode >/dev/null
    if cmp -s opre_\$mode.bin opost_\$mode.bin; then echo \"  ORACLE mode=\$mode : IDENTICAL (0 byte mismatch)\"; else echo \"  ORACLE mode=\$mode : *** MISMATCH ***\"; gate_ok=0; fi
  done
  [ \$gate_ok -eq 1 ] || { echo '  byte-exact gate FAILED — POST is not the golden result; aborting timing'; exit 41; }

  echo '=== [3/4] A/B + vs-opponent (cold N=$N rounds; each round pre then post as fresh procs) ==='
  echo '    KQGEMM line: ours_gmacs / opp_gmacs / ratio_ours_over_opp'
  for i in \$(seq 1 $N); do
    r_pre=\$(  LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./kqgemm_pre  q4_K $K $NR $NC $ITERS 0xC0FFEE )
    echo \"ROUND \$i pre  \$r_pre\"
    r_post=\$( LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./kqgemm_post q4_K $K $NR $NC $ITERS 0xC0FFEE )
    echo \"ROUND \$i post \$r_post\"
  done
"
echo "[reroll-ab] done. restore: main tree + build/ UNTOUCHED (no rebuild); board scratch ephemeral under $RDIR; governor left as-found."
