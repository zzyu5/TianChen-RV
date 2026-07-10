#!/usr/bin/env bash
# [CASE-COMPILER-ASYMMETRY Stage-1 rvv] Phase C: self-check SOP (measurement gate, NOT loadavg gate).
# 3x remeasure-remeasure of one representative binary (bin_q2K_gcc); compute spread% of the
# ours-side best_gmacs (historical cv 0.12-0.5%) and ratio_med. Gate = <= micro-floor x1.5 = 1.2%.
set -uo pipefail
RDIR=/tmp/case_stage1_rvv; CORE=8; GGML=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
cd "$RDIR"
echo "[selfcheck] loadavg=$(cat /proc/loadavg)"
run(){ LD_LIBRARY_PATH=$GGML taskset -c $CORE ./bin_q2K_gcc q2_K 2048 64 512 12 0xC0FFEE 2>/dev/null; }
for i in 1 2 3; do
  L=$(run)
  og=$(echo "$L"|grep -oE 'OURS best_ns=[0-9]+ med_ns=[0-9]+ best_gmacs=[0-9.]+'|grep -oE 'best_gmacs=[0-9.]+'|cut -d= -f2)
  rm=$(echo "$L"|grep -oE 'ratio_med=[0-9.]+'|cut -d= -f2)
  rb=$(echo "$L"|grep -oE 'ratio_best=[0-9.]+'|cut -d= -f2)
  echo "SC$i ours_best_gmacs=$og ratio_best=$rb ratio_med=$rm"
done
echo "[selfcheck-loadavg-end] $(cat /proc/loadavg)"
