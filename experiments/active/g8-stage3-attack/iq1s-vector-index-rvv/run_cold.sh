#!/usr/bin/env bash
# G8 §六.3 iq1_s/iq1_m cold A/B census run. arg1=tag arg2=outfile
set -uo pipefail
source /opt/tcrv-toolchains/env.sh 2>/dev/null
GCCLIB=/opt/tcrv-toolchains/gcc-15.2.0/lib64
WD=/tmp/g8s3iq/rvv; cd $WD
TAG=${1:-baseline}; OUT=${2:-cold_${TAG}.log}
CORE=8
: > $OUT
echo "# G8 §六.3 iq cold A/B  TAG=$TAG  $(date -u)" | tee -a $OUT
echo "# loadavg_begin=$(cat /proc/loadavg)" | tee -a $OUT
echo "# core$CORE gov=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null)" | tee -a $OUT
echo "# co-tenant check (vLLM cores 0,1): $(ps -eo pid,psr,comm 2>/dev/null | awk '$2==0||$2==1' | grep -iE 'vllm|python' | head -3 | tr '\n' ';')" | tee -a $OUT
run(){ LD_LIBRARY_PATH=$GCCLIB taskset -c $CORE ./iq_bench "$@" 2>>$OUT | tee -a $OUT; }
for f in iq1_s iq1_m; do
  for nc in 64 256 512; do
    for rep in 1 2; do
      SEED=$(printf '0x%X' $((0x1000 + RANDOM)))
      run $f 2048 1 $nc 8 15 $SEED
    done
  done
done
echo "# loadavg_end=$(cat /proc/loadavg)" | tee -a $OUT
echo "# ALL_DONE" | tee -a $OUT
