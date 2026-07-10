#!/usr/bin/env bash
# [CASE-COMPILER-ASYMMETRY Stage-1 rvv] Phase D: full symmetric remeasure, N cold interleaved rounds.
# 6 binaries: {q2K,q5T3,q5L1} x {gcc=SYMMETRIC, clang=REFERENCE}. Opponent block-dot = gcc-15 .so.
# Interleave all 6 per round (drift-cancel). Report per-binary ours/opp gmacs + ratio each round.
set -uo pipefail
RDIR=/tmp/case_stage1_rvv; CORE=8; N=${N:-14}
GGML=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
cd "$RDIR"
echo "[measure] N=$N core=$CORE  loadavg-start=$(cat /proc/loadavg)"
echo "  bin -> {fmt,args}: q2K=q2_K 2048/64/512 ; q5T3=q5_K 2048/64/512 ; q5L1=q5_K 2048/64/512"
declare -A FMT=( [q2K]=q2_K [q5T3]=q5_K [q5L1]=q5_K )
run(){ # $1=bintag  -> emits: <tag> ours_gmacs opp_gmacs ratio_best ratio_med
  local tag=$1 f=${FMT[${1%%_*}]}
  local L; L=$(LD_LIBRARY_PATH=$GGML taskset -c $CORE ./bin_$tag $f 2048 64 512 20 0xC0FFEE 2>/dev/null)
  local og opp rb rm
  og=$(echo "$L"|grep -oE 'OURS best_ns=[0-9]+ med_ns=[0-9]+ best_gmacs=[0-9.]+'|grep -oE 'best_gmacs=[0-9.]+'|cut -d= -f2)
  opp=$(echo "$L"|grep -oE 'OPP best_ns=[0-9]+ med_ns=[0-9]+ best_gmacs=[0-9.]+'|grep -oE 'best_gmacs=[0-9.]+'|cut -d= -f2)
  rb=$(echo "$L"|grep -oE 'ratio_best=[0-9.]+'|cut -d= -f2)
  rm=$(echo "$L"|grep -oE 'ratio_med=[0-9.]+'|cut -d= -f2)
  echo "$tag ours=$og opp=$opp ratio_best=$rb ratio_med=$rm"
}
for i in $(seq 1 $N); do
  echo "== ROUND $i =="
  for b in q2K_gcc q2K_clang q5T3_gcc q5T3_clang q5L1_gcc q5L1_clang; do
    echo "R$i $(run $b)"
  done
done
echo "[measure] loadavg-end=$(cat /proc/loadavg)"
echo "=== [Stage-1 measure DONE] ==="
