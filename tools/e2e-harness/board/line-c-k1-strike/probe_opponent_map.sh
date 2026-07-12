#!/usr/bin/env bash
# Line C · k1 精准打击 — opponent_map + q8_0 VLEN256 health probe (READ-ONLY, board CLEAN).
# Reproduces experiments/active/line-c-k1-strike/raw_probe_evidence.txt. No board mutation, no timing.
# Usage: ssh k1 'bash -s' < probe_opponent_map.sh    (or scp then run)
set -u
STOCK=/data/k1build-stock/bin/libggml-cpu.so.0
SRC=/home/bianbu/tcrv-k1-llama/ggml/src/ggml-cpu/repack.cpp   # read-only reference (Line B tree; DO NOT WRITE)

echo "== [P0] board fingerprint =="
grep -m1 isa /proc/cpuinfo; nproc; uptime

echo "== [P1] stock compiler identity (CMakeCache) =="
grep -E "CMAKE_C(XX)?_COMPILER:" /data/k1build-stock/CMakeCache.txt
readelf -p .comment "$STOCK" 2>/dev/null | sort -u | grep -iE "clang|GCC"

echo "== [P2] SpacemiT/IME vendor path in stock? (expect 0) =="
nm -D "$STOCK" 2>/dev/null | grep -ic spacemit

echo "== [P3] riscv-16x1 tuned repack traits present (as-shipped tier-A) =="
nm -D "$STOCK" 2>/dev/null | grep -E "gemm_.*16x1|gemv_.*16x1" | grep -v generic | sort

echo "== [P4] q8_0 (+q4_K/q2_K/iq4_nl) 16x1 vl-handling: vsetivli 16 == VLMAX@256 = HEALTHY =="
for s in ggml_gemm_q8_0_16x1_q8_0 ggml_gemv_q8_0_16x1_q8_0 ggml_gemm_q4_K_16x1_q8_K \
         ggml_gemm_q2_K_16x1_q8_K ggml_gemm_iq4_nl_16x1_q8_0; do
  echo "-- $s --"
  objdump -d --disassemble="$s" "$STOCK" 2>/dev/null | grep -iE "vsetivli|vsetvli" | head -2
done

echo "== [P5] block-dot vec_dot opponents (tier-B fallback) vectorization =="
for s in ggml_vec_dot_q5_K_q8_K ggml_vec_dot_q6_K_q8_K ggml_vec_dot_q3_K_q8_K; do
  n=$(objdump -d --disassemble="$s" "$STOCK" 2>/dev/null | grep -cE $'\tv[a-z]')
  v=$(objdump -d --disassemble="$s" "$STOCK" 2>/dev/null | grep -cE "vsetvli|vsetivli")
  echo "  $s : rvv_insns=$n vsetvli=$v"
done

echo "== [P3-src] dispatch selector per format (q5_K/q6_K -> nullptr -> block-dot) =="
sed -n '4528,4760p' "$SRC" | grep -nE "type == GGML_TYPE_Q[0-9]_[K01]|riscv_v|return &|return nullptr|case 256" | head -40

echo "== board CLEAN: no scratch created by this probe (read-only) =="
