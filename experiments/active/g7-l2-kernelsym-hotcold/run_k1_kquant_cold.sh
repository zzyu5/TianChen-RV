#!/usr/bin/env bash
# run_k1_kquant_cold.sh — G7 L2 货架A cold micro A/B for q4_K + q5_K @k1/VLEN256.
# Measures HOT (best-of-5, single buffer, existing driver) AND COLD (pool>>LLC, median-of-N)
# in the SAME session on the SAME core, so the hot/cold delta is clean (no cross-session confound).
# Opponent = board's own /data/k1build-stock libggml-cpu.so ggml_vec_dot_q{4,5}_K_q8_K (SAME as t4a hot).
# [NG-4] kernel-axis datapoint, NOT e2e, NOT a sealed Win. Ours=clang-18; opp=factory-as-shipped.
set -euo pipefail
K="${K:-2048}"; NC="${NC:-512}"; CORE="${CORE:-3}"
POOL="${POOL:-8}"; ROUNDS="${ROUNDS:-12}"; HITERS="${HITERS:-10}"; HREPS="${HREPS:-3}"
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"; CC="${CC:-clang}"
GGML_BIN="${GGML_BIN:-/data/k1build-stock/bin}"
SRC="${SRC:-/tmp/tcrv_k1_kquant_t4a}"     # has s6_q4K.c, gemm_q5_K_q8_K.kernel.c, kquant_gemm_paired_driver.c
SCR="${SCR:-/tmp/g7_l2_hotcold}"
mkdir -p "$SCR"; cd "$SCR"

echo "==== G7-L2 cold hot/cold  K=$K nc=$NC core=$CORE POOL=$POOL ROUNDS=$ROUNDS ===="
echo "---- [0] PREFLIGHT + LOAD-GATE ----"
echo "board=k1 uname=$(uname -srm)"; echo "cc=$($CC --version | head -1)"
echo "march=$MARCH mabi=lp64d ffp-contract=on -O2 (clang-18 symmetric domain)"
echo "governor(core$CORE)=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null||echo NA) freq=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null||echo NA)"
echo "L2(core$CORE)=$(cat /sys/devices/system/cpu/cpu$CORE/cache/index2/size 2>/dev/null) shared=$(cat /sys/devices/system/cpu/cpu$CORE/cache/index2/shared_cpu_list 2>/dev/null)"
LA1=$(cut -d' ' -f1 /proc/loadavg); echo "loadavg_begin=$(cat /proc/loadavg)"
# load-gate: warn (not abort) — k1 is shared; paired within-proc ratio cancels common contention
awk -v la="$LA1" 'BEGIN{ if(la>5.0) print "LOAD-GATE: loadavg "la" > 5.0 HIGH (ratios still paired; noted)"; else print "LOAD-GATE: loadavg "la" OK" }'
echo "opponent_lib=$GGML_BIN/libggml-cpu.so"
nm -D "$GGML_BIN/libggml-cpu.so" | grep -E 'T ggml_vec_dot_q[45]_K_q8_K$' | sed 's/^/opp_sym /'
echo "md5 ours emitted C (proves reuse, no regen):"; md5sum "$SRC/s6_q4K.c" "$SRC/gemm_q5_K_q8_K.kernel.c" | sed 's/^/  /'

echo "---- [1] COMPILE (ours clang-18 -O2; opponent linked from factory .so) ----"
$CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ "$SRC/s6_q4K.c"               -c -o s6_O2.o
$CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ "$SRC/gemm_q5_K_q8_K.kernel.c" -c -o kq5k_O2.o
for o in s6_O2.o kq5k_O2.o; do objdump -d $o 2>/dev/null | grep -Eq '__extendhfsf2|__truncsfhf2|__gnu_h2f_ieee|__gnu_f2h_ieee' && echo "  $o SOFT-FP16 LIBCALL" || echo "  $o libcall-free OK"; done
# HOT driver (existing) + COLD driver (new)
$CC -O2 -march=$MARCH -mabi=lp64d -x c "$SRC/kquant_gemm_paired_driver.c" -c -o hotdrv.o
$CC hotdrv.o s6_O2.o kq5k_O2.o -o kqgemm_hot  -L"$GGML_BIN" -Wl,-rpath,"$GGML_BIN" -lggml-cpu -lggml-base -lggml -lm
$CC -O2 -march=$MARCH -mabi=lp64d -x c ./kquant_gemm_cold_driver.c -c -o colddrv.o
$CC colddrv.o s6_O2.o kq5k_O2.o -o kqgemm_cold -L"$GGML_BIN" -Wl,-rpath,"$GGML_BIN" -lggml-cpu -lggml-base -lggml -lm
echo "  built kqgemm_hot + kqgemm_cold"

run_hot(){  # fmt nr iters
  LD_LIBRARY_PATH="$GGML_BIN" taskset -c $CORE ./kqgemm_hot "$1" $K "$2" $NC "$3" 0xC0FFEE
}
run_cold(){ # fmt nr pool rounds
  LD_LIBRARY_PATH="$GGML_BIN" taskset -c $CORE ./kqgemm_cold "$1" $K "$2" $NC "$3" "$4" 0xC0FFEE
}

echo "---- [2] HOT (best-of-5, single buffer, nr=64 prefill; sanity vs sealed 3.106x/1.916x) ----"
for fmt in q4_K q5_K; do
  for i in $(seq 1 $HREPS); do
    echo "HOT $fmt rep$i :: $(run_hot $fmt 64 $HITERS)"
  done
done

echo "---- [3] COLD nr=64 (hot-shape, pool>>LLC; isolates cache-residency at prefill shape) ----"
for fmt in q4_K q5_K; do
  echo "COLD64 $fmt :: $(run_cold $fmt 64 $POOL $ROUNDS)"
done

echo "---- [4] COLD nr=4 (decode-leaning min-nr, pool>>LLC; memory-bound e2e-decode predictor) ----"
for fmt in q4_K q5_K; do
  echo "COLD4 $fmt :: $(run_cold $fmt 4 $POOL $ROUNDS)"
done

echo "loadavg_end=$(cat /proc/loadavg)"
echo "==== DONE (scratch $SCR; main tree + build untouched; governor left as-found) ===="
