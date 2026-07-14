#!/usr/bin/env bash
# run_q6k_roll_ab.sh — G7 L1' q6_K@rvv GEMM whole-K-nest roll G2 board-validate.
# Symmetric gcc-15.2 (ours + opponent libggml-cpu.so both gcc-15). Board rvv/VLEN128.
# Byte-exact A/B (rolled vs unrolled mismatch=0) + 3-way cold GEMM prefill timing.
# Core-pinned within reserved band 8-15; load-gated; NO co-tenant disturbance; no git.
set -uo pipefail
CORE="${CORE:-8}"
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"
GGML_BIN="/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin"
RDIR="/tmp/g7_q6k_roll_ab"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[q6k-roll-ab] core=$CORE march=$MARCH"
ssh rvv "mkdir -p $RDIR"
scp -q "$HERE/q6k_roll_ab_driver.c" "$HERE/k_q6_unrolled.cpp" "$HERE/k_q6_rolled.cpp" rvv:$RDIR/

ssh rvv "set -uo pipefail; cd $RDIR
  # ---- LOAD-GATE: abort if the board is busy (protect co-tenants + timing) ----
  L1=\$(cut -d' ' -f1 /proc/loadavg)
  echo '# loadavg_begin='\$(cat /proc/loadavg)' users='\$(who|wc -l)
  awk -v l=\$L1 'BEGIN{ if(l+0>12.0){ print \"[ABORT] loadavg \"l\" > 12 (board busy)\"; exit 41 } }' || exit 41

  source /opt/tcrv-toolchains/env.sh 2>/dev/null
  GCC=\$(command -v gcc); echo '[cc] '\$(\$GCC --version|head -1)' | opp_lib=gcc-15'
  echo '# board='\$(uname -srm)' core='$CORE' gov='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null)' freq='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null)
  ls $GGML_BIN/libggml-cpu.so >/dev/null || { echo GGML_MISSING; exit 40; }
  echo '# ggml_so_md5='\$(md5sum $GGML_BIN/libggml-cpu.so|cut -d' ' -f1)

  echo '=== [A] compile both q6_K emitters gcc-15.2 -O2 + objdump static account ==='
  for tag in unrolled rolled; do
    \$GCC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ k_q6_\${tag}.cpp -c -o k_\${tag}.o 2>cc_\${tag}.err || { echo KERN_FAIL \$tag; sed -n '1,8p' cc_\${tag}.err; exit 3; }
    VS=\$(objdump -d k_\${tag}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
    SP=\$(objdump -d k_\${tag}.o 2>/dev/null | grep -cE 'vs[1248]r\.v')
    RL=\$(objdump -d k_\${tag}.o 2>/dev/null | grep -cE 'vl[1248]re[0-9]+\.v')
    WM=\$(objdump -d k_\${tag}.o 2>/dev/null | grep -cE 'vwmacc')
    MV=\$(objdump -d k_\${tag}.o 2>/dev/null | grep -oE '[ ,(]v[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
    echo \"# Q6_\${tag}(gcc-15 .o): vsetvli=\$VS spill=\$SP reload=\$RL vwmacc=\$WM maxVreg=v\$MV size=\$(wc -c<k_\${tag}.o)B\"
  done

  echo '=== [B] link driver + both kernels + real libggml-cpu.so ==='
  \$GCC -O2 -march=$MARCH -mabi=lp64d -x c q6k_roll_ab_driver.c -c -o drv.o 2>cc_drv.err || { echo DRV_CC_FAIL; sed -n '1,10p' cc_drv.err; exit 4; }
  \$GCC drv.o k_unrolled.o k_rolled.o -L$GGML_BIN -Wl,-rpath,$GGML_BIN -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o q6kab 2>ld.err || { echo LINK_FAIL; sed -n '1,12p' ld.err; exit 5; }
  echo '  linked OK -> ./q6kab'

  echo '=== [C] BYTE-EXACT A/B + 3-way COLD GEMM prefill (core '$CORE', pinned) ==='
  echo '# loadavg_pre='\$(cat /proc/loadavg)
  # prefill shape: M=nr>=4 (GEMM), several nc/K; N reps>=12 cold
  for cfg in '2048 4 512' '2048 8 512' '2048 16 512' '4096 8 256'; do
    set -- \$cfg; K=\$1; NR=\$2; NC=\$3
    SEED=\$(printf '0x%X' \$((0x1234 + NR*7 + NC)))
    LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./q6kab \$K \$NR \$NC 8 12 \$SEED
  done
  echo '# loadavg_end='\$(cat /proc/loadavg)
  # ---- cleanup: no stray procs ----
  pkill -u \$(id -un) -f q6kab 2>/dev/null; rm -f q6kab drv.o k_unrolled.o k_rolled.o
  echo '=== DONE (cleaned) ==='
"
