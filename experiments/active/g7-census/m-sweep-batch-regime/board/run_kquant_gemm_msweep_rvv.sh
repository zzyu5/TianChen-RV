#!/usr/bin/env bash
# run_kquant_gemm_msweep_rvv.sh — G7 §3 M-sweep batch-regime: REPACK-GEMM @rvv M-sweep.
# ours = weft-emitted repack-GEMM (reads weight ONCE, reuses across nr=M rows => arithmetic
#   intensity ∝ M). opp = as-shipped ggml_vec_dot block-dot replicated over M×nc (re-streams
#   weight per row => intensity flat). This is the compute-bound crossover probe: if ours' GMAC/s
#   is flat in M it is already compute-bound at M=4; the ratio vs opp climbs iff opp degrades.
# nr(M) grid {4,8,16,32} (repack-GEMM packs activations x4 => nr%4==0; M=1 GEVM is the vecdot lane).
# Symmetric gcc-15.2, VLEN128, HOT best-of + COLD 224MiB flush paired median. Reuses SEALED
# kquant_gemm_census_driver.c + kernels from ../../batch1-kquant-rvv. NO git, main tree untouched.
set -uo pipefail
SRC="/home/kingdom/phdworks/TianchenRV/experiments/active/g7-census/batch1-kquant-rvv"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDIR=/tmp/g7_msweep_kquant_rvv
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"
GGML="/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin"
K="${K:-2048}"; NC="${NC:-512}"; HITERS="${HITERS:-6}"; REPS="${REPS:-12}"
NRGRID="${NRGRID:-4 8 16 32}"
FMTS="q2 q3 q4 q5 q6"

echo "[msweep-kquant-rvv] scp harness -> rvv:$RDIR (nr grid=$NRGRID)"
ssh rvv "mkdir -p $RDIR"
scp -q "$SRC/kquant_gemm_census_driver.c" rvv:$RDIR/
for f in $FMTS; do scp -q "$SRC/kernels/repack_gemm_${f}_K_q8_K.kernel.c" rvv:$RDIR/; done

ssh rvv "set -uo pipefail; cd $RDIR
  L1=\$(cut -d' ' -f1 /proc/loadavg)
  awk -v l=\$L1 'BEGIN{ if(l+0>12.0){ print \"[ABORT] loadavg \"l\" > 12\"; exit 41 } }' || exit 41
  source /opt/tcrv-toolchains/env.sh 2>/dev/null
  GCC=\$(command -v gcc)
  LOG=$RDIR/run.log; : > \$LOG
  echo '# BUILD kquant-gemm-msweep@rvv CC='\$(\$GCC --version|head -1)' march=$MARCH' | tee -a \$LOG
  echo '# board='\$(uname -srm)' ggml_cpu_md5='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$LOG
  ls $GGML/libggml-cpu.so >/dev/null || { echo GGML_MISSING; exit 40; }
  OBJS=''
  for f in $FMTS; do
    \$GCC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ repack_gemm_\${f}_K_q8_K.kernel.c -c -o k_\${f}.o 2>cc_\${f}.err || { echo KERN_FAIL \$f; sed -n '1,6p' cc_\${f}.err; exit 3; }
    OBJS=\"\$OBJS k_\${f}.o\"
  done
  \$GCC -O2 -march=$MARCH -mabi=lp64d -x c kquant_gemm_census_driver.c -c -o drv.o 2>cc_drv.err || { echo DRV_CC_FAIL; sed -n '1,10p' cc_drv.err; exit 4; }
  \$GCC drv.o \$OBJS -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o census 2>ld.err || { echo LINK_FAIL; sed -n '1,12p' ld.err; exit 5; }
  echo '# linked OK -> ./census' | tee -a \$LOG

  # ---- load-gate: pick idle core in 8-15 (co-tenant vLLM on 0,1) ----
  echo '# loadavg_begin='\$(cat /proc/loadavg) | tee -a \$LOG
  read_busy(){ awk -v c=\"cpu\$1\" '\$1==c{idle=\$5+\$6; tot=\$2+\$3+\$4+\$5+\$6+\$7+\$8; print tot\" \"idle}' /proc/stat; }
  declare -A B0 I0
  for c in 8 9 10 11 12 13 14 15; do read t i < <(read_busy \$c); B0[\$c]=\$t; I0[\$c]=\$i; done
  sleep 0.4
  BESTC=-1; BESTIDLE=-1
  for c in 8 9 10 11 12 13 14 15; do read t i < <(read_busy \$c); dt=\$((t-\${B0[\$c]})); di=\$((i-\${I0[\$c]}));
    pct=\$(( dt>0 ? 100*di/dt : 0 )); if [ \$pct -gt \$BESTIDLE ]; then BESTIDLE=\$pct; BESTC=\$c; fi; done
  if [ \$BESTIDLE -lt 70 ]; then echo \"# LOAD_GATE_FAIL best core\$BESTC idle=\${BESTIDLE}% (<70) ABORT\" | tee -a \$LOG; exit 9; fi
  CORE=\$BESTC
  echo \"# LOAD_GATE_OK pin core=\$CORE idle=\${BESTIDLE}% gov=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_governor 2>/dev/null) freq=\$(cat /sys/devices/system/cpu/cpu\${CORE}/cpufreq/scaling_cur_freq 2>/dev/null)\" | tee -a \$LOG
  run(){ LD_LIBRARY_PATH=$GGML taskset -c \$CORE ./census \"\$@\" 2>>\$LOG | tee -a \$LOG; }

  echo '=== [S] REPACK-GEMM M-sweep: 5 fmt x nr{$NRGRID} (K=$K nc=$NC hiters=$HITERS reps=$REPS) ===' | tee -a \$LOG
  for nr in $NRGRID; do
    for fmt in q2_K q3_K q4_K q5_K q6_K; do
      SEED=\$(printf '0x%X' \$((0x1234 + nr*7)))
      run \$fmt $K \$nr $NC $HITERS $REPS \$SEED
    done
    echo \"# loadavg_after_nr\${nr}=\"\$(cat /proc/loadavg) | tee -a \$LOG
  done
  echo '# ggml_cpu_md5_after='\$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1) | tee -a \$LOG
  pkill -u \$(id -un) -f census 2>/dev/null
  echo '# STRAY='\$(pgrep -c -f '/census' || echo 0) | tee -a \$LOG
  echo '# loadavg_end='\$(cat /proc/loadavg) | tee -a \$LOG
  echo '# ALL_DONE' | tee -a \$LOG
"
mkdir -p "$HERE/raw"
scp -q rvv:$RDIR/run.log "$HERE/raw/rvv_kquant_gemm_msweep.log" 2>/dev/null && echo "[msweep-kquant-rvv] log pulled"
