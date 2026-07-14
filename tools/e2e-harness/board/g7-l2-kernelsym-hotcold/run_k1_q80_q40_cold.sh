#!/usr/bin/env bash
# run_k1_q80_q40_cold.sh — G7 L2 货架A hot/cold micro A/B for q8_0 GEVM + q4_0 GEMM @k1/VLEN256.
# Opponent = board /data/k1build-stock libggml-cpu.so ggml_vec_dot_q{8_0,4_0}_q8_0 block-dot.
# [NG-4] kernel-axis, NOT e2e, NOT sealed Win. Ours=clang-18; opp=factory gcc-15.
set -euo pipefail
K="${K:-2048}"; CORE="${CORE:-3}"; POOL="${POOL:-8}"; ROUNDS="${ROUNDS:-12}"; HITERS="${HITERS:-20}"
NC_GEVM="${NC_GEVM:-2048}"; NC_GEMM="${NC_GEMM:-512}"
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"; CC="${CC:-clang}"
GGML_BIN="${GGML_BIN:-/data/k1build-stock/bin}"
SCR="${SCR:-/tmp/g7_l2_hotcold}"; cd "$SCR"

echo "==== G7-L2 q8_0/q4_0 hot/cold  K=$K core=$CORE POOL=$POOL ROUNDS=$ROUNDS ===="
echo "board=k1 uname=$(uname -srm)"; echo "cc=$($CC --version|head -1)"
echo "march=$MARCH -O2 (clang-18 symmetric domain)"
echo "governor(core$CORE)=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null||echo NA) freq=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null||echo NA)"
echo "L2(core$CORE)=$(cat /sys/devices/system/cpu/cpu$CORE/cache/index2/size 2>/dev/null)"
LA1=$(cut -d' ' -f1 /proc/loadavg); echo "loadavg_begin=$(cat /proc/loadavg)"
awk -v la="$LA1" 'BEGIN{ print (la>5.0)? "LOAD-GATE: "la" HIGH (paired ratio noted)":"LOAD-GATE: "la" OK" }'
nm -D "$GGML_BIN/libggml-cpu.so" | grep -E 'T ggml_vec_dot_q(8_0|4_0)_q8_0$' | sed 's/^/opp_sym /'
echo "md5 ours kernels:"; md5sum q8_0_gevm_wide_hl16.kernel.c q4_0_repack_gemm.kernel.c | sed 's/^/  /'

echo "---- COMPILE ----"
$CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ q8_0_gevm_wide_hl16.kernel.c -c -o q80.o
$CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ q4_0_repack_gemm.kernel.c    -c -o q40.o
for o in q80.o q40.o; do objdump -d $o 2>/dev/null | grep -Eq '__extendhfsf2|__truncsfhf2|__gnu_h2f_ieee|__gnu_f2h_ieee' && echo "  $o SOFT-FP16 LIBCALL" || echo "  $o libcall-free OK"; done
$CC -O2 -march=$MARCH -mabi=lp64d -x c gevm_q8_cold_driver.c  -c -o q80drv.o
$CC -O2 -march=$MARCH -mabi=lp64d -x c gemm_q4_0_cold_driver.c -c -o q40drv.o
$CC q80drv.o q80.o -o q80cold -L"$GGML_BIN" -Wl,-rpath,"$GGML_BIN" -lggml-cpu -lggml-base -lggml -lm
$CC q40drv.o q40.o -o q40cold -L"$GGML_BIN" -Wl,-rpath,"$GGML_BIN" -lggml-cpu -lggml-base -lggml -lm
echo "  built q80cold + q40cold"

echo "---- [q8_0 GEVM decode-native nr=1] nc=$NC_GEVM ----"
LD_LIBRARY_PATH="$GGML_BIN" taskset -c $CORE ./q80cold $K $NC_GEVM $HITERS $POOL $ROUNDS 0xC0FFEE

echo "---- [q4_0 GEMM nr=64 prefill/hot-shape] nc=$NC_GEMM ----"
LD_LIBRARY_PATH="$GGML_BIN" taskset -c $CORE ./q40cold $K 64 $NC_GEMM $HITERS $POOL $ROUNDS 0xC0FFEE
echo "---- [q4_0 GEMM nr=4 decode-leaning] nc=$NC_GEMM ----"
LD_LIBRARY_PATH="$GGML_BIN" taskset -c $CORE ./q40cold $K 4 $NC_GEMM $HITERS $POOL $ROUNDS 0xC0FFEE

echo "loadavg_end=$(cat /proc/loadavg)"
echo "==== DONE ===="
