#!/usr/bin/env bash
# run_rvv_kquant_census.sh — G7 L1 Batch-1: K-quant@rvv cold-start census (5 formats).
# Symmetric gcc-15.2 (both ours kernel + opponent libggml-cpu.so built gcc-15). Board rvv/VLEN128.
# HOT (best-of-N single buffer) + COLD (224MiB flush paired A/B, median N) same session, core-pinned.
# Correctness proven separately (per-format repack verifiers). Main tree + build/ UNTOUCHED; no git.
#
# Usage: [MODE=smoke|full] [CORE=8] [K=2048] [NC=512] [HITERS=8] [REPS=12] bash run_rvv_kquant_census.sh
set -uo pipefail
MODE="${MODE:-full}"; CORE="${CORE:-8}"; K="${K:-2048}"; NC="${NC:-512}"; HITERS="${HITERS:-8}"; REPS="${REPS:-12}"
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"
GGML_BIN="/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin"
RDIR="/tmp/g7_census_kquant_rvv"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[census] MODE=$MODE core=$CORE K=$K nc=$NC hiters=$HITERS reps=$REPS march=$MARCH"
ssh rvv "mkdir -p $RDIR"
scp -q "$HERE/kquant_gemm_census_driver.c" rvv:$RDIR/
for f in q2 q3 q4 q5 q6; do scp -q "$HERE/kernels/repack_gemm_${f}_K_q8_K.kernel.c" rvv:$RDIR/; done

ssh rvv "set -uo pipefail; cd $RDIR
  source /opt/tcrv-toolchains/env.sh 2>/dev/null
  GCC=\$(command -v gcc); echo '[cc] '\$(\$GCC --version|head -1)' | opp_lib=gcc-15'
  echo '# board='\$(uname -srm) ' core='$CORE ' gov='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null)' freq='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null)
  echo '# ggml_so_md5='\$(md5sum $GGML_BIN/libggml-cpu.so|cut -d' ' -f1)
  echo '# loadavg_begin='\$(cat /proc/loadavg)
  ls $GGML_BIN/libggml-cpu.so >/dev/null || { echo GGML_MISSING; exit 40; }

  echo '=== [A] compile ours kernels gcc-15.2 -O2 (libcall-free check) ==='
  OBJS=''
  for f in q2 q3 q4 q5 q6; do
    \$GCC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ repack_gemm_\${f}_K_q8_K.kernel.c -c -o k_\${f}.o 2>cc_\${f}.err || { echo KERN_FAIL \$f; sed -n '1,6p' cc_\${f}.err; exit 3; }
    LC=\$(objdump -d k_\${f}.o 2>/dev/null | grep -Eqc '__truncsfhf2|__extendhfsf2|__gnu_f2h_ieee|__gnu_h2f_ieee' && echo SOFTFP || echo cleanfp)
    VS=\$(objdump -d k_\${f}.o 2>/dev/null | grep -cE 'vsetvli|vsetivli')
    SP=\$(objdump -d k_\${f}.o 2>/dev/null | grep -cE 'vs[1248]r\.v')
    RL=\$(objdump -d k_\${f}.o 2>/dev/null | grep -cE 'vl[1248]r\.v')
    WM=\$(objdump -d k_\${f}.o 2>/dev/null | grep -cE 'vwmacc')
    MV=\$(objdump -d k_\${f}.o 2>/dev/null | grep -oE '[ ,(]v[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
    echo \"# OURS_\${f}_K(gcc-15 .o): fp16=\$LC vsetvli=\$VS spill=\$SP reload=\$RL vwmacc=\$WM maxVreg=v\$MV size=\$(wc -c<k_\${f}.o)B\"
    OBJS=\"\$OBJS k_\${f}.o\"
  done

  echo '=== [B] opponent symbol machine-probe (which kernel dispatches @VLEN128?) ==='
  for q in q2 q3 q4 q5 q6; do
    MAIN=\$(nm -D $GGML_BIN/libggml-cpu.so 2>/dev/null | grep -E \" T ggml_vec_dot_\${q}_K_q8_K\$\" | head -1 | awk '{print \$1}')
    V128=\$(nm -D $GGML_BIN/libggml-cpu.so 2>/dev/null | grep -E \" T ggml_vec_dot_\${q}_K_q8_K_vl128\$\" | head -1 | awk '{print \$1}')
    GEMM=\$(nm -D $GGML_BIN/libggml-cpu.so 2>/dev/null | grep -E \" T ggml_gemm_\${q}_K_(16x1|8x8)_q8_K(_generic)?\$\" | awk '{print \$2}' | tr '\n' ',')
    echo \"# OPP_\${q}_K: main=\${MAIN:-ABSENT} vl128=\${V128:-none} unused_repack_gemm=\${GEMM:-none}\"
  done

  echo '=== [C] link driver + ours + real libggml-cpu.so ==='
  \$GCC -O2 -march=$MARCH -mabi=lp64d -x c kquant_gemm_census_driver.c -c -o drv.o 2>cc_drv.err || { echo DRV_CC_FAIL; sed -n '1,10p' cc_drv.err; exit 4; }
  \$GCC drv.o \$OBJS -L$GGML_BIN -Wl,-rpath,$GGML_BIN -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o census 2>ld.err || { echo LINK_FAIL; sed -n '1,12p' ld.err; exit 5; }
  echo '  linked OK -> ./census'

  if [ '$MODE' = smoke ]; then
    echo '=== [SMOKE] q4_K K=512 nr=16 (sanity: links, runs, ratio finite) ==='
    LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./census q4_K 512 16 512 4 10 0xC0FFEE
    echo '=== [SMOKE] q6_K K=512 nr=16 ==='
    LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./census q6_K 512 16 512 4 10 0xC0FFEE
    echo '# loadavg_end='\$(cat /proc/loadavg); exit 0
  fi

  echo '=== [D] CENSUS sweep: 5 fmt x nr{4,8,16,64} (K=$K nc=$NC hiters=$HITERS reps=$REPS core $CORE) ==='
  echo '# loadavg_pre_sweep='\$(cat /proc/loadavg)
  for fmt in q2_K q3_K q4_K q5_K q6_K; do
    for nr in 4 8 16 64; do
      SEED=\$(printf '0x%X' \$((0x1234 + nr*7)))
      LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./census \$fmt $K \$nr $NC $HITERS $REPS \$SEED
    done
    echo \"# loadavg_after_\${fmt}=\"\$(cat /proc/loadavg)
  done
  echo '# loadavg_end='\$(cat /proc/loadavg)
  echo '=== DONE ==='
"
