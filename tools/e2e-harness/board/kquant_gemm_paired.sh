#!/usr/bin/env bash
# kquant_gemm_paired.sh — [KQUANT-L1] prefill paired A/B (ours repack GEMM vs opponent real
# block-dot) for q4_K + q5_K on rvv/VLEN128. Compiles OUR exported kernels + the paired driver
# with clang-17 and links the board's OWN libggml-cpu.so so the opponent side is ggml's REAL
# dispatched vl128 block-dot (ggml_vec_dot_q{4,5}_K_q8_K). Main tree + build/ UNTOUCHED; nothing
# committed; no git stash. [NG-4]: L1 candidate datapoint, NOT a beat (eight gates not walked).
#
# Usage: BOARD=rvv CORE=8 K=2048 NR=16 NC=512 ITERS=20 bash tools/e2e-harness/board/kquant_gemm_paired.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD="${BOARD:-rvv}"; CORE="${CORE:-8}"; K="${K:-2048}"; NR="${NR:-16}"; NC="${NC:-512}"; ITERS="${ITERS:-20}"
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"; CC="${CC:-clang-17}"
GGML_BIN="${GGML_BIN:-/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin}"
RDIR="/tmp/kquant_export"

echo "[kqgemm] board=$BOARD core=$CORE K=$K nr=$NR nc=$NC iters=$ITERS march=$MARCH ggml_bin=$GGML_BIN"
ssh "$BOARD" "mkdir -p $RDIR"
scp -q "$HERE/kquant_gemm_paired_driver.c" "$BOARD:$RDIR/"

ssh "$BOARD" "set -e; cd $RDIR
  echo '[preflight] freq='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null || echo NA)' gov='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null || echo NA)
  ls $GGML_BIN/libggml-cpu.so >/dev/null || { echo 'GGML_BIN missing libggml-cpu.so'; exit 40; }
  echo '[compile] our kernels + driver + link real libggml-cpu.so'
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ gemm_q4_K_q8_K.kernel.c -c -o kq4k.o
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ gemm_q5_K_q8_K.kernel.c -c -o kq5k.o
  $CC -O2 -march=$MARCH -mabi=lp64d -x c kquant_gemm_paired_driver.c -c -o kqdrv.o
  $CC kqdrv.o kq4k.o kq5k.o -o kqgemm -L$GGML_BIN -Wl,-rpath,$GGML_BIN -lggml-cpu -lggml-base -lggml -lm
  echo '[preflight] libcall-free (our kernels):'
  for o in kq4k.o kq5k.o; do (llvm-objdump-17 -d \$o 2>/dev/null||objdump -d \$o 2>/dev/null) | grep -Eq '__truncsfhf2|__extendhfsf2|__gnu_f2h_ieee|__gnu_h2f_ieee' && echo \"  \$o SOFT-FP16 LIBCALL\" || echo \"  \$o libcall-free OK\"; done
  echo '=== KQUANT-L1 PREFILL PAIRED (best-of-5, VLEN128) ==='
  LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./kqgemm q4_K $K $NR $NC $ITERS 0xC0FFEE
  LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./kqgemm q5_K $K $NR $NC $ITERS 0xC0FFEE
"
echo "[kqgemm] done. restore: nothing to restore (main tree + build/ untouched; scratch under $RDIR on board)."
