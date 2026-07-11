#!/usr/bin/env bash
# stream_paired_board.sh -- RUNS ON THE BOARD (rvv). Compiles the streaming paired driver +
# the exported OUR kernels twice (clang deploy ledger / gcc kernel-symmetric ledger), links the
# board's own ggml .so as the opponent, and runs all covering-batch③ streaming cells both ways.
# Board scratch only; main tree + build/ untouched. No git.
set -u
SCRATCH=${SCRATCH:-/tmp/stream_batch3}
GGML=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
CORE=${CORE:-40}
MARCH="-march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d -ffp-contract=on -O3"
N=${N:-25165824}      # 24*1024*1024 elems (mult of 256) -> ~96MiB float side, cold
ROUNDS=${ROUNDS:-15}
cd "$SCRATCH" || exit 9

echo "## board $(hostname) $(date -u +%FT%TZ) core=$CORE N=$N rounds=$ROUNDS"
echo "## ggml opponent md5: $(md5sum $GGML/libggml-cpu.so.0 | awk '{print $1}') (cpu) / $(md5sum $GGML/libggml-base.so.0 | awk '{print $1}') (base)"
echo "## clang: $(clang --version|head -1) ; gcc: $(gcc --version|head -1)"

# emitted kernels use fabsf/memset without including math.h/string.h -> prepend (idempotent)
for k in deq_q8_0 deq_q4_K deq_iq4_nl qnt_q8_0 qnt_q8_K fwd_add fwd_mul fwd_gelu; do
  if ! head -1 $k.c | grep -q '__STREAM_HDRS__'; then
    printf '// __STREAM_HDRS__\n#include <math.h>\n#include <string.h>\n' | cat - $k.c > $k.c.tmp && mv $k.c.tmp $k.c
  fi
done

build_ledger(){  # $1 = clang|gcc  -> ONLY the OUR-kernel .o codegen varies; driver+link ALWAYS g++ (neutral,
                 # = the compiler that built the opponent .so) per [CASE-COMPILER-ASYMMETRY] flat recipe.
  local tag=$1 KCXX
  if [ "$tag" = clang ]; then KCXX=clang++; else KCXX=g++; fi
  local objs=""
  for k in deq_q8_0 deq_q4_K deq_iq4_nl qnt_q8_0 qnt_q8_K fwd_add fwd_mul fwd_gelu; do
    $KCXX $MARCH -c $k.c -o ${k}.${tag}.o 2>build_${k}_${tag}.err || { echo "COMPILE-FAIL $k $tag"; cat build_${k}_${tag}.err; return 1; }
    objs="$objs ${k}.${tag}.o"
  done
  # driver + FINAL LINK always g++-15 (neutral): only kernel codegen differs between ledgers
  g++ $MARCH -x c++ -c stream_paired_driver.c -o driver.${tag}.o 2>build_driver_${tag}.err \
     || { echo "DRIVER-COMPILE-FAIL $tag"; cat build_driver_${tag}.err; return 1; }
  g++ $MARCH driver.${tag}.o $objs -o driver_${tag} \
       -L$GGML -lggml-base -lggml-cpu -lggml -Wl,-rpath,$GGML -lm 2>link_${tag}.err \
     || { echo "LINK-FAIL $tag"; cat link_${tag}.err; return 1; }
  echo "built driver_${tag} (kernels=$KCXX, driver+link=g++-15)"
}

run_cell(){  # $1=binary $2=ledger $3=mode $4=fmt
  LD_LIBRARY_PATH=$GGML taskset -c $CORE ./$1 $3 $4 $N $ROUNDS 0x51EA \
    | sed "s/^STREAM/RESULT ledger=$2/"
}

build_ledger clang || exit 1
build_ledger gcc   || exit 1

echo "### self-check (3x same cell, deq q8_0, clang) -- IQR baseline"
for i in 1 2 3; do run_cell driver_clang clang dequant q8_0; done

echo "### DEQUANT reps (q8_0 flat / q4_K super-block / iq4_nl codebook)"
for fmt in q8_0 q4_K iq4_nl; do
  run_cell driver_clang clang dequant $fmt
  run_cell driver_gcc   gcc   dequant $fmt
done
echo "### QUANT reps (q8_0 / q8_K)"
for fmt in q8_0 q8_K; do
  run_cell driver_clang clang quant $fmt
  run_cell driver_gcc   gcc   quant $fmt
done
echo "### FORWARD reps (add / mul vector-map ; gelu scalar-map[LUT-caveat])"
for fmt in add mul gelu; do
  run_cell driver_clang clang forward $fmt
  run_cell driver_gcc   gcc   forward $fmt
done
echo "## DONE"
