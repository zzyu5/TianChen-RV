#!/usr/bin/env bash
# run_k1_q4k_vl16_hb.sh — G7 §L1 ★G decisive: q4_K@k1 SEALED vl=16 core (s6_q4K_vl16_sealed.c md5 e437fd3b)
#   vs TRUE shipped hand-brick ggml_gemm_q4_K_16x1_q8_K (case256 VLEN256-native repack).
# This is the vl=16 counterpart of the vl=8 handbrick-resolve (which lost 0.622x). The sealed vl=16
#   core is what won e2e 1.085x; here we do the kernel-axis MICRO A/B never before directly measured.
# Symmetric: BOTH clang-18. Opponent = arch/riscv/repack.cpp.o in /data/k1build-stock.
# We compile OURS both at opponent's exact recipe (-O3 fair-march) AND our canonical -O2.
# NO git · NO schema/T8/ROADMAP · NO rvv · board scratch only · main tree/build/governor UNTOUCHED.
set -euo pipefail
K="${K:-2048}"; NR="${NR:-64}"; NC="${NC:-512}"; POOL="${POOL:-8}"; ROUNDS="${ROUNDS:-12}"; ITERS="${ITERS:-20}"; CORE="${CORE:-3}"
GGML_BIN="${GGML_BIN:-/data/k1build-stock/bin}"
CXX="${CXX:-clang++-18}"; CC="${CC:-clang-18}"
MARCH_OPP="rv64gcv_zfh_zvfh_zicbop_zihintpause"   # opponent's exact march (stock repack.cpp.o)
MARCH_CANON="rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs"    # our t4a canonical march
KSRC="s6_q4K_vl16_sealed.c"

echo "==== G DECISIVE: q4_K@k1 SEALED vl=16 vs TRUE hand-brick 16x1  K=$K nr=$NR nc=$NC pool=$POOL rounds=$ROUNDS iters=$ITERS core=$CORE ===="
echo "---- [0] PREFLIGHT ----"
echo "board=k1 uname=$(uname -srm)"
echo "cxx=$($CXX --version | head -1)"
echo "opponent build recipe (from compile_commands.json) = clang++-18 -O3 -march=$MARCH_OPP -mabi=lp64d (arch/riscv/repack.cpp)"
{ printf '#include <riscv_vector.h>\n#include <stdio.h>\nint main(void){long b=(long)__riscv_vlenb();printf("vlenb=%%ld VLEN_bits=%%ld\\n",b,b*8);return 0;}\n' > _vb.c; $CC -O2 -march=$MARCH_OPP -mabi=lp64d _vb.c -o _vb && ./_vb; }
echo "governor(core$CORE)=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null||echo NA) freq=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null||echo NA)"
echo "opponent_lib=$GGML_BIN/libggml-cpu.so -> $(readlink -f $GGML_BIN/libggml-cpu.so)"
nm -D "$GGML_BIN/libggml-cpu.so" | grep -E 'T ggml_gemm_q4_K_16x1_q8_K$' | sed 's/^/  opp_sym /'
echo "md5 our emitted kernel (must == e437fd3b..., proves NO regen, == sealed vl=16 core):"
md5sum $KSRC | sed 's/^/  /'

echo "---- [1] COMPILE (ours clang-18; two recipes; opponent linked from stock lib) ----"
$CXX -O3 -march=$MARCH_OPP   -mabi=lp64d -ffp-contract=on -std=gnu++20 -x c++ $KSRC -c -o ours_O3.o
$CXX -O2 -march=$MARCH_CANON -mabi=lp64d -ffp-contract=on -x c++         $KSRC -c -o ours_O2.o
$CC  -O2 -march=$MARCH_OPP   -mabi=lp64d -x c q4k_handbrick_driver.c -c -o drv.o
$CXX drv.o ours_O3.o -o kqhb_O3 -L"$GGML_BIN" -Wl,-rpath,"$GGML_BIN" -lggml-cpu -lggml-base -lggml -lm
$CXX drv.o ours_O2.o -o kqhb_O2 -L"$GGML_BIN" -Wl,-rpath,"$GGML_BIN" -lggml-cpu -lggml-base -lggml -lm
echo "[libcall-free check ours]:"
for o in ours_O3.o ours_O2.o; do objdump -d $o 2>/dev/null | grep -Eq '__truncsfhf2|__extendhfsf2|__gnu_f2h_ieee|__gnu_h2f_ieee' && echo "  $o SOFT-FP16 LIBCALL" || echo "  $o libcall-free OK"; done

echo "---- [2] OBJDUMP SEAL (ours vl=16 vs opponent structure) ----"
seal(){ local DIS; DIS=$(objdump -d "$1" 2>/dev/null)
  local vs=$( printf '%s' "$DIS" | grep -cE 'vsetvli|vsetivli' || true )
  local sp=$( printf '%s' "$DIS" | grep -cE 'vs[1248]r\.v'     || true )
  local wm=$( printf '%s' "$DIS" | grep -cE 'vwmacc'           || true )
  local txhex=$( objdump -h "$1" 2>/dev/null | awk '$2==".text"{print $3}' ); local tx=$(( 16#${txhex:-0} ))
  printf '  OBJD %-12s vsetvli=%-4s spill=%-4s vwmacc=%-5s textB=%s\n' "$2" "$vs" "$sp" "$wm" "$tx"
}
seal ours_O3.o 'OURS-vl16 -O3'; seal ours_O2.o 'OURS-vl16 -O2'
echo "  OPP  ggml_gemm_q4_K_16x1_q8_K (stock, from lib objdump): vsetvli~34 spill~14 vwmacc~80 (VLEN256-native e32m2/e16m1/e8mf2/m4)"

echo "---- [3] LOAD-GATE (until 1min loadavg < 2.5 or 40 polls) ----"
for i in $(seq 1 40); do la=$(cut -d' ' -f1 /proc/loadavg); ok=$(awk -v l="$la" 'BEGIN{print (l<2.5)?1:0}'); echo "  poll $i loadavg1=$la ok=$ok"; [ "$ok" = "1" ] && break; sleep 3; done
echo "loadavg_begin=$(cat /proc/loadavg)"

echo "---- [4] A/B RUN (ours vl=16 vs TRUE 16x1 hand-brick; correctness + HOT + COLD; N=$ROUNDS) ----"
for tag in O3 O2; do
  echo "===== OURS-vl16-$tag (vs 16x1) ====="
  for s in 0xC0FFEE 0xBEEF01 0x51A7ED; do
    echo "--- seed $s ---"
    LD_LIBRARY_PATH="$GGML_BIN" taskset -c $CORE ./kqhb_$tag $K $NR $NC $POOL $ROUNDS $ITERS $s
  done
done
echo "loadavg_end=$(cat /proc/loadavg)"
echo "==== DONE (board scratch $(pwd); main tree/build/governor UNTOUCHED; stock lib read-only) ===="
