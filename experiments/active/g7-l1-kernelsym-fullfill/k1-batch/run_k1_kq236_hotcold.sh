#!/usr/bin/env bash
# run_k1_kq236_hotcold.sh — G7 §L1 k1-half: q2_K/q3_K/q6_K @k1/VLEN256 kernel-sym hot/cold A/B.
# OURS = tcrv_emitc repack GEMM PLAIN exports (kq_export_q6q2q3). OPP = stock block-dot
#   ggml_vec_dot_q{2,3,6}_K_q8_K (machine-judged public T symbol). Symmetric clang-18.
# NO git · NO schema/T8/ROADMAP · NO rvv · board scratch only · main tree/build/governor UNTOUCHED.
set -euo pipefail
K="${K:-2048}"; NC="${NC:-512}"; POOL="${POOL:-8}"; ROUNDS="${ROUNDS:-12}"; ITERS="${ITERS:-20}"; CORE="${CORE:-3}"
NRS="${NRS:-64 16 4}"
GGML_BIN="${GGML_BIN:-/data/k1build-stock/bin}"
CXX="${CXX:-clang++-18}"; CC="${CC:-clang-18}"
MARCH_CANON="rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs"   # our t4a canonical march
MARCH_OPP="rv64gcv_zfh_zvfh_zicbop_zihintpause"  # opponent's exact march

echo "==== G7 L1 k1-half q2/q3/q6 kernel-sym hot/cold  K=$K nc=$NC pool=$POOL rounds=$ROUNDS iters=$ITERS core=$CORE nrs='$NRS' ===="
echo "---- [0] PREFLIGHT ----"
echo "board=k1 uname=$(uname -srm)"; echo "cxx=$($CXX --version | head -1)"
{ printf '#include <riscv_vector.h>\n#include <stdio.h>\nint main(void){long b=(long)__riscv_vlenb();printf("vlenb=%%ld VLEN_bits=%%ld\\n",b,b*8);return 0;}\n' > _vb.c; $CC -O2 -march=$MARCH_CANON -mabi=lp64d _vb.c -o _vb && ./_vb; }
echo "governor(core$CORE)=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null||echo NA) freq=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null||echo NA)"
echo "opponent_lib=$GGML_BIN/libggml-cpu.so -> $(readlink -f $GGML_BIN/libggml-cpu.so)"
echo "opponent symbols (machine-judged, public T):"
nm -D "$GGML_BIN/libggml-cpu.so" | grep -E 'T ggml_vec_dot_q[236]_K_q8_K$' | sed 's/^/  opp_sym /'
echo "md5 our emitted kernels (must match kq_export: q2=6c9322f6 q3=f4cfb641 q6=0f14791e):"
md5sum repack_gemm_q2_K_q8_K.kernel.c repack_gemm_q3_K_q8_K.kernel.c repack_gemm_q6_K_q8_K.kernel.c | sed 's/^/  /'

echo "---- [1] COMPILE ours 3 kernels (clang-18 -O2 canonical) + driver, link one binary ----"
for f in q2_K q3_K q6_K; do
  echo "  compiling ours $f ..."
  $CXX -O2 -march=$MARCH_CANON -mabi=lp64d -ffp-contract=on -x c++ repack_gemm_${f}_q8_K.kernel.c -c -o ours_${f}.o
done
$CC -O2 -march=$MARCH_OPP -mabi=lp64d -x c kquant_gemm_hotcold_q236_driver.c -c -o drv.o
$CXX drv.o ours_q2_K.o ours_q3_K.o ours_q6_K.o -o kq236 -L"$GGML_BIN" -Wl,-rpath,"$GGML_BIN" -lggml-cpu -lggml-base -lggml -lm
echo "[libcall-free check ours]:"
for f in q2_K q3_K q6_K; do objdump -d ours_${f}.o 2>/dev/null | grep -Eq '__truncsfhf2|__extendhfsf2|__gnu_f2h_ieee|__gnu_h2f_ieee' && echo "  ours_${f}.o SOFT-FP16 LIBCALL" || echo "  ours_${f}.o libcall-free OK"; done

echo "---- [2] OBJDUMP SEAL (deployment-compiler spill axis; G1 精化②) ----"
seal(){ local DIS; DIS=$(objdump -d "$1" 2>/dev/null)
  local vs=$( printf '%s' "$DIS" | grep -cE 'vsetvli|vsetivli' || true )
  local sp=$( printf '%s' "$DIS" | grep -cE 'vs[1248]r\.v'     || true )
  local wm=$( printf '%s' "$DIS" | grep -cE 'vwmacc'           || true )
  local txhex=$( objdump -h "$1" 2>/dev/null | awk '$2==".text"{print $3}' ); local tx=$(( 16#${txhex:-0} ))
  printf '  OBJD %-14s vsetvli=%-5s spill=%-5s vwmacc=%-6s textB=%s\n' "$2" "$vs" "$sp" "$wm" "$tx"
}
seal ours_q2_K.o 'OURS q2_K -O2'; seal ours_q3_K.o 'OURS q3_K -O2'; seal ours_q6_K.o 'OURS q6_K -O2'

echo "---- [3] LOAD-GATE (until 1min loadavg < 2.5 or 40 polls) ----"
for i in $(seq 1 40); do la=$(cut -d' ' -f1 /proc/loadavg); ok=$(awk -v l="$la" 'BEGIN{print (l<2.5)?1:0}'); echo "  poll $i loadavg1=$la ok=$ok"; [ "$ok" = "1" ] && break; sleep 3; done
echo "loadavg_begin=$(cat /proc/loadavg)"

echo "---- [4] A/B RUN (hot+cold; fmt x nr x 3 seeds; N=$ROUNDS) ----"
for fmt in q2_K q3_K q6_K; do
  for NR in $NRS; do
    echo "===== $fmt nr=$NR ====="
    for s in 0xC0FFEE 0xBEEF01 0x51A7ED; do
      echo "--- $fmt nr=$NR seed $s ---"
      LD_LIBRARY_PATH="$GGML_BIN" taskset -c $CORE ./kq236 $fmt $K $NR $NC $POOL $ROUNDS $ITERS $s
    done
  done
done
echo "loadavg_end=$(cat /proc/loadavg)"
echo "==== DONE (board scratch $(pwd); main tree/build/governor UNTOUCHED; stock lib read-only) ===="
