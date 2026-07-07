#!/usr/bin/env bash
# kquant_gemm_pipeline_ab.sh — [KQUANT-L1 / pipeline] q4_K repack GEMM PRE-pipelining vs
# POST-pipelining A/B on rvv/VLEN128, plus vs-opponent block-dot. Isolates whether the
# byte-exact decode-hoist schedule rearrangement (the "pipelining" tracer bullet) converts the
# structural register-pressure opening into throughput.
#
#   PRE  = golden emitter lowering  (/tmp/q4k_emitc_golden.mlir  -> pre_q4K.c ; == cached baseline)
#   POST = decode-hoist rearranged  (/tmp/q4k_emitc_after.mlir   -> post_q4K.c ; op-multiset ==)
#
# Both compiled clang-17 -O2 (SAME flags as kquant_gemm_paired.sh) into SEPARATE binaries and
# linked against the board's OWN libggml-cpu.so (opponent = real dispatched block-dot). Two board
# objdumps (pre/post kernel .o) record the actual measured-binary vsetvli count. Cold N rounds:
# each round runs pre then post as FRESH processes (cold caches). Main tree + build/ UNTOUCHED;
# nothing committed; no git stash. [NG-4]: L1 pipeline datapoint, NOT a beat (eight gates unwalked).
#
# Usage: BOARD=rvv CORE=8 K=2048 NR=64 NC=512 ITERS=20 N=12 bash tools/e2e-harness/board/kquant_gemm_pipeline_ab.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOARD="${BOARD:-rvv}"; CORE="${CORE:-8}"; K="${K:-2048}"; NR="${NR:-64}"; NC="${NC:-512}"
ITERS="${ITERS:-20}"; N="${N:-12}"
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"; CC="${CC:-clang-17}"
GGML_BIN="${GGML_BIN:-/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin}"
SRC="${SRC:-/tmp/pl_ab}"          # host dir holding pre_q4K.c / post_q4K.c / gemm_q5_K_q8_K.kernel.c
RDIR="/tmp/kquant_pipeline_ab"

echo "[pl-ab] board=$BOARD core=$CORE K=$K nr=$NR nc=$NC iters=$ITERS N=$N march=$MARCH"
ssh "$BOARD" "mkdir -p $RDIR"
scp -q "$SRC/pre_q4K.c" "$SRC/post_q4K.c" "$SRC/gemm_q5_K_q8_K.kernel.c" \
       "$HERE/kquant_gemm_paired_driver.c" "$BOARD:$RDIR/"

ssh "$BOARD" "set -e; cd $RDIR
  echo '[preflight] freq='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null || echo NA)' gov='\$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null || echo NA)' vlenb='\$(cat /proc/cpuinfo >/dev/null; echo)
  ls $GGML_BIN/libggml-cpu.so >/dev/null || { echo 'GGML_BIN missing libggml-cpu.so'; exit 40; }
  echo '[compile] PRE + POST q4_K kernels + shared q5_K + driver (clang-17 -O2)'
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ pre_q4K.c  -c -o pre_q4k.o
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ post_q4K.c -c -o post_q4k.o
  $CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ gemm_q5_K_q8_K.kernel.c -c -o kq5k.o
  $CC -O2 -march=$MARCH -mabi=lp64d -x c kquant_gemm_paired_driver.c -c -o kqdrv.o
  $CC kqdrv.o pre_q4k.o  kq5k.o -o kqgemm_pre  -L$GGML_BIN -Wl,-rpath,$GGML_BIN -lggml-cpu -lggml-base -lggml -lm
  $CC kqdrv.o post_q4k.o kq5k.o -o kqgemm_post -L$GGML_BIN -Wl,-rpath,$GGML_BIN -lggml-cpu -lggml-base -lggml -lm
  echo '[objdump] measured-binary vsetvli (clang-17 -O2, the ACTUAL board form):'
  for o in pre_q4k.o post_q4k.o; do
    DIS=\$( (llvm-objdump-17 -d \$o 2>/dev/null||objdump -d \$o 2>/dev/null) )
    nv=\$(  printf '%s' \"\$DIS\" | grep -cE 'vsetvli|vsetivli' || true )
    nvv=\$( printf '%s' \"\$DIS\" | grep -cE 'vsetvli' || true )
    nwm=\$( printf '%s' \"\$DIS\" | grep -cE 'vwmacc' || true )
    lc=\$(  printf '%s' \"\$DIS\" | grep -cE '__truncsfhf2|__extendhfsf2|__gnu_f2h_ieee|__gnu_h2f_ieee' || true )
    echo \"  OBJD \$o vset_all=\$nv vsetvli=\$nvv vwmacc=\$nwm fp16libcall=\$lc\"
  done
  echo '=== PIPELINE A/B (cold N=$N rounds, each round pre then post as fresh procs) ==='
  for i in \$(seq 1 $N); do
    r_pre=\$( LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./kqgemm_pre  q4_K $K $NR $NC $ITERS 0xC0FFEE )
    echo \"ROUND \$i pre  \$r_pre\"
    r_post=\$( LD_LIBRARY_PATH=$GGML_BIN taskset -c $CORE ./kqgemm_post q4_K $K $NR $NC $ITERS 0xC0FFEE )
    echo \"ROUND \$i post \$r_post\"
  done
"
echo "[pl-ab] done. restore: nothing to restore (main tree + build/ untouched; scratch under $RDIR on board)."
