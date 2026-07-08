#!/usr/bin/env bash
# kquant_gemm_tile_s6_ab.sh — [KQUANT-L1 / T2-tile S6] q4_K repack GEMM
#   PRE (S1 h-strip tile, committed d5a28efc) vs POST (S6 inline-min-fold + stack-panel) A/B on
#   rvv/VLEN128, plus vs-opponent block-dot parity. S6 is the register-cliff step of the 6-step tile
#   ladder: stage the idle-across-the-hot-dot i32 MIN accumulator (bsums) + the 6-bit scale/min decode
#   strips + on-demand d/dmin to STACK PANELS so the hot main-term dot's peak live-vreg fits in <=32
#   and allocator live-value spills go to ~0 — byte-exact-preserving (single f32 convert+fold kept in
#   the SAME order; i32 accumulation reordered but exact+associative). Siblings: l1-pipeline (#1 NULL),
#   l1-reroll (#2 -11%), l1-tile-s1 (#3 +53%, spill 84->23). This S6 = step 6, logged ALONE (not merged S1).
#
#   golden = golden_q4K.c  (full-unroll, md5 b0b5beac, oracle-GREEN @039133ea)  -- objdump 84-base anchor ONLY
#   PRE    = s1_q4K.c       (S1 h-strip tile, md5 9a757c02; committed d5a28efc)  -- A/B baseline + timed
#   POST   = s6_q4K.c       (S6 min-fold+stack-panel, md5 90d454da; working-tree delta)  -- timed
#
#   Measures: (1) objdump vsetvli + spill(vs{1,2,4,8}r.v)/reload(vl{1,2,4,8}r.v) + maxVreg golden->S1->S6
#   at the MEASURED board form (clang-17 -O2) AND the diagnosis form (-O3) — does spill reach ~0, peak <=32?
#   (2) byte-exact identity gate (int + norm) PRE(S1) vs POST(S6) via `cmp` (correctness of the timed S6).
#   (3) A/B ours_post(S6)/ours_pre(S1) paired cold N rounds (S6-isolated incremental speedup over S1).
#   (4) vs-opponent parity ours/opp for S1 and S6 (S1 was 1.471x; does S6 go higher?).
#
#   Both q4_K kernels compiled clang-17 -O2 (SAME flags as kquant_gemm_tile_s1_ab.sh) into SEPARATE
#   binaries, each linked against the board's OWN libggml-cpu.so (opponent = real dispatched block-dot).
#   Cold N rounds: each round runs pre(S1) then post(S6) as FRESH processes. Main tree + build/ UNTOUCHED;
#   nothing committed; no git stash (S1 baseline is the cached export). [NG-4]: L1 datapoint, NOT a beat.
#
# Usage: BOARD=rvv CORE=8 K=2048 NR=64 NC=512 ITERS=20 N=12 bash tools/e2e-harness/board/kquant_gemm_tile_s6_ab.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD="${BOARD:-rvv}"; CORE="${CORE:-8}"; K="${K:-2048}"; NR="${NR:-64}"; NC="${NC:-512}"
ITERS="${ITERS:-20}"; N="${N:-12}"
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"; CC="${CC:-clang-17}"
GGML_BIN="${GGML_BIN:-/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin}"
SRC="${SRC:-/tmp/q4k_tile_s6}"    # host dir: golden_q4K.c s1_q4K.c s6_q4K.c gemm_q5_K_q8_K.kernel.c verify_q4k_identity.c
RDIR="/tmp/kquant_tile_s6_ab"

echo "[tile-s6-ab] board=$BOARD core=$CORE K=$K nr=$NR nc=$NC iters=$ITERS N=$N march=$MARCH cc=$CC"
ssh "$BOARD" "mkdir -p $RDIR"
scp -q "$SRC/golden_q4K.c" "$SRC/s1_q4K.c" "$SRC/s6_q4K.c" "$SRC/gemm_q5_K_q8_K.kernel.c" "$SRC/verify_q4k_identity.c" \
       "$HERE/kquant_gemm_paired_driver.c" "$BOARD:$RDIR/"

ssh "$BOARD" "set -e; cd $RDIR
  echo '[preflight] loadavg='\$(cat /proc/loadavg)' core=$CORE freq='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null||echo NA)' gov='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null||echo NA)
  ls $GGML_BIN/libggml-cpu.so >/dev/null || { echo 'GGML_BIN missing libggml-cpu.so'; exit 40; }

  echo '[compile] golden/S1/S6 q4_K at -O2 (measured) and -O3 (diagnosis); shared q5_K + paired driver (-O2)'
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ golden_q4K.c -c -o gold_O2.o
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ s1_q4K.c    -c -o s1_O2.o
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ s6_q4K.c    -c -o s6_O2.o
  $CC -O3 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ golden_q4K.c -c -o gold_O3.o
  $CC -O3 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ s1_q4K.c    -c -o s1_O3.o
  $CC -O3 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ s6_q4K.c    -c -o s6_O3.o
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ gemm_q5_K_q8_K.kernel.c -c -o kq5k.o
  $CC -O2 -march=$MARCH -mabi=lp64d -x c kquant_gemm_paired_driver.c -c -o kqdrv.o
  $CC kqdrv.o s1_O2.o kq5k.o -o kqgemm_pre  -L$GGML_BIN -Wl,-rpath,$GGML_BIN -lggml-cpu -lggml-base -lggml -lm
  $CC kqdrv.o s6_O2.o kq5k.o -o kqgemm_post -L$GGML_BIN -Wl,-rpath,$GGML_BIN -lggml-cpu -lggml-base -lggml -lm

  echo '=== [1] OBJDUMP seal: vsetvli + spill/reload + maxVreg  golden->S1->S6 (spill ~0? peak <=32?) ==='
  seal(){  # \$1=objfile \$2=label
    local DIS; DIS=\$(objdump -d \$1 2>/dev/null)
    local vs=\$(  printf '%s' \"\$DIS\" | grep -cE 'vsetvli|vsetivli' || true )
    local sp=\$(  printf '%s' \"\$DIS\" | grep -cE 'vs[1248]r\\.v'    || true )
    local rl=\$(  printf '%s' \"\$DIS\" | grep -cE 'vl[1248]r\\.v'    || true )
    local wm=\$(  printf '%s' \"\$DIS\" | grep -cE 'vwmacc'           || true )
    local mv=\$(  printf '%s' \"\$DIS\" | grep -oE '[ ,(]v[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1 )
    local tx=\$(  objdump -h \$1 2>/dev/null | awk '/\\.text/{print strtonum(\"0x\"\$3)}' )
    printf '  OBJD %-11s vsetvli=%-4s spill=%-4s reload=%-4s vwmacc=%-5s maxVreg=v%-3s textB=%s\\n' \"\$2\" \"\$vs\" \"\$sp\" \"\$rl\" \"\$wm\" \"\$mv\" \"\$tx\"
  }
  seal gold_O2.o 'GOLD -O2'; seal s1_O2.o 'S1   -O2'; seal s6_O2.o 'S6   -O2'
  seal gold_O3.o 'GOLD -O3'; seal s1_O3.o 'S1   -O3'; seal s6_O3.o 'S6   -O3'

  echo '=== [2] BYTE-EXACT identity gate PRE(S1) vs POST(S6) (int + norm) ==='
  $CC -O2 -march=$MARCH -mabi=lp64d -x c verify_q4k_identity.c -c -o vdrv.o
  $CC vdrv.o s1_O2.o -o verify_pre  -lm
  $CC vdrv.o s6_O2.o -o verify_post -lm
  gate_ok=1
  for mode in int norm; do
    taskset -c $CORE ./verify_pre  \$mode $K $NR $NC 0xC0FFEE opre_\$mode  >/dev/null
    taskset -c $CORE ./verify_post \$mode $K $NR $NC 0xC0FFEE opost_\$mode >/dev/null
    if cmp -s opre_\$mode.bin opost_\$mode.bin; then echo \"  ORACLE mode=\$mode : IDENTICAL (0 byte mismatch)\"; else echo \"  ORACLE mode=\$mode : *** MISMATCH ***\"; gate_ok=0; fi
  done
  [ \$gate_ok -eq 1 ] || { echo '  byte-exact gate FAILED — POST(S6) is not the S1/golden result; aborting timing'; exit 41; }

  echo '=== [3/4] A/B(S6 vs S1) + vs-opponent (cold N=$N rounds; each round pre(S1) then post(S6) as fresh procs) ==='
  echo '    KQGEMM line: ours_gmacs / opp_gmacs / ratio_ours_over_opp'
  echo '[loadavg-mid] '\$(cat /proc/loadavg)
  for i in \$(seq 1 $N); do
    r_pre=\$(  LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./kqgemm_pre  q4_K $K $NR $NC $ITERS 0xC0FFEE )
    echo \"ROUND \$i pre  \$r_pre\"
    r_post=\$( LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./kqgemm_post q4_K $K $NR $NC $ITERS 0xC0FFEE )
    echo \"ROUND \$i post \$r_post\"
  done
  echo '[loadavg-end] '\$(cat /proc/loadavg)
"
echo "[tile-s6-ab] done. restore: main tree + build/ UNTOUCHED (no rebuild); board scratch ephemeral under $RDIR; governor left as-found."
