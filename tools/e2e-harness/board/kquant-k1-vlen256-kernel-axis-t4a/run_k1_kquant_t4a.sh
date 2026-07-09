#!/usr/bin/env bash
# run_k1_kquant_t4a.sh — [SEL-1 T4a] k1/VLEN256 kernel-axis board batch for q4_K (+q5_K).
# BOARD-SIDE runner: expects the cached emitted C already in CWD (NO local tcrv-opt regen):
#   golden_q4K.c (md5 b0b5beac) · s6_q4K.c (md5 90d454da, S6 tiled q4_K) ·
#   gemm_q5_K_q8_K.kernel.c (md5 ba30ba54, q5_K) · verify_q4k_identity.c · kquant_gemm_paired_driver.c
# Closes gate ③ (k1 objdump seal, VLEN256/LMUL codegen) + gate ⑤ (k1 kernel-axis paired vs factory block-dot).
# Opponent = board's OWN libggml-cpu.so dispatched ggml_vec_dot_q{4,5}_K_q8_K (VLEN256 branch via __riscv_vlenb).
# [NG-4]: kernel-axis L1 datapoint, NOT an e2e beat. Ours=clang-18; opp=factory-as-shipped (same asymmetry as rvv).
set -euo pipefail
K="${K:-2048}"; NR="${NR:-64}"; NC="${NC:-512}"; ITERS="${ITERS:-20}"; N="${N:-12}"; CORE="${CORE:-7}"
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"; CC="${CC:-clang}"
GGML_BIN="${GGML_BIN:-/data/k1build-stock/bin}"

echo "==== T4a k1 kernel-axis  K=$K nr=$NR nc=$NC iters=$ITERS N=$N core=$CORE ===="
echo "---- [0] PREFLIGHT (board identity + symmetry) ----"
echo "board=k1 uname=$(uname -srm)"
echo "cc=$($CC --version | head -1)"
echo "march=$MARCH  mabi=lp64d  ffp-contract=on (measured -O2)"
{ printf '#include <riscv_vector.h>\n#include <stdio.h>\nint main(void){long b=(long)__riscv_vlenb();printf("vlenb=%%ld VLEN_bits=%%ld\\n",b,b*8);return 0;}\n' > _vb.c; $CC -O2 -march=$MARCH -mabi=lp64d _vb.c -o _vb && ./_vb; }
echo "governor(core$CORE)=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null||echo NA) freq=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null||echo NA)"
echo "loadavg_begin=$(cat /proc/loadavg)"
echo "opponent_lib=$GGML_BIN/libggml-cpu.so"
ls "$GGML_BIN/libggml-cpu.so" >/dev/null || { echo "GGML_BIN missing libggml-cpu.so"; exit 40; }
nm -D "$GGML_BIN/libggml-cpu.so" | grep -E 'T ggml_vec_dot_q[45]_K_q8_K$' | sed 's/^/opp_sym /'
echo "md5 emitted C (must match host cache, proves NO regen):"
md5sum golden_q4K.c s6_q4K.c gemm_q5_K_q8_K.kernel.c | sed 's/^/  /'

echo "---- [1] COMPILE (ours clang-18 -O2 measured + -O3 diagnosis; opponent linked from factory lib) ----"
$CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ golden_q4K.c            -c -o gold_O2.o
$CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ s6_q4K.c               -c -o s6_O2.o
$CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ gemm_q5_K_q8_K.kernel.c -c -o kq5k_O2.o
$CC -O3 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ golden_q4K.c            -c -o gold_O3.o
$CC -O3 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ s6_q4K.c               -c -o s6_O3.o
$CC -O2 -march=$MARCH -mabi=lp64d -x c kquant_gemm_paired_driver.c               -c -o kqdrv.o
$CC kqdrv.o s6_O2.o kq5k_O2.o -o kqgemm -L"$GGML_BIN" -Wl,-rpath,"$GGML_BIN" -lggml-cpu -lggml-base -lggml -lm
echo "[libcall-free check ours kernels]:"
for o in s6_O2.o kq5k_O2.o; do objdump -d $o 2>/dev/null | grep -Eq '__truncsfhf2|__extendhfsf2|__gnu_f2h_ieee|__gnu_h2f_ieee' && echo "  $o SOFT-FP16 LIBCALL" || echo "  $o libcall-free OK"; done

echo "---- [2] OBJDUMP SEAL (VLEN256/LMUL codegen; seal method == rvv campaign-canonical) ----"
seal(){ local DIS; DIS=$(objdump -d "$1" 2>/dev/null)
  local vs=$( printf '%s' "$DIS" | grep -cE 'vsetvli|vsetivli' || true )
  local sp=$( printf '%s' "$DIS" | grep -cE 'vs[1248]r\.v'     || true )
  local rl=$( printf '%s' "$DIS" | grep -cE 'vl[1248]r\.v'     || true )
  local wm=$( printf '%s' "$DIS" | grep -cE 'vwmacc'           || true )
  local mv=$( printf '%s' "$DIS" | grep -oE '[ ,(]v[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1 )
  local txhex=$( objdump -h "$1" 2>/dev/null | awk '$2==".text"{print $3}' ); local tx=$(( 16#${txhex:-0} ))
  printf '  OBJD %-9s vsetvli=%-4s spill=%-4s reload=%-4s vwmacc=%-5s maxVreg=v%-3s textB=%s\n' "$2" "$vs" "$sp" "$rl" "$wm" "$mv" "$tx"
}
seal gold_O2.o 'GOLD -O2'; seal s6_O2.o 'S6   -O2'; seal kq5k_O2.o 'Q5K  -O2'
seal gold_O3.o 'GOLD -O3'; seal s6_O3.o 'S6   -O3'
echo "[vsetvli SEW/LMUL descriptors in S6 -O2 (VLA vtype selection at VLEN256):]"
objdump -d s6_O2.o 2>/dev/null | grep -oE 'vsetvli[^#]*e[0-9]+,\s*m[f0-9]+' | sed -E 's/.*(e[0-9]+),\s*(m[f0-9]+).*/\1,\2/' | sort | uniq -c | sort -rn | sed 's/^/  /'
echo "[full seal objdump -> objdump_k1_s6_seal.objdump]"
{ echo "# k1/VLEN256 (SpacemiT X60) q4_K S6 seal, $($CC --version|head -1) -O2 & -O3, march=$MARCH"
  echo "# vlenb=32 (VLEN256). seal method == rvv canonical (vs[1248]r.v spill / vl[1248]r.v reload / vwmacc / maxVreg)."
  seal gold_O2.o 'GOLD -O2'; seal s6_O2.o 'S6   -O2'; seal kq5k_O2.o 'Q5K  -O2'
  seal gold_O3.o 'GOLD -O3'; seal s6_O3.o 'S6   -O3'
  echo "# --- S6 -O2 hot-region vsetvli/vwmacc excerpt (first 60) ---"
  objdump -d s6_O2.o 2>/dev/null | grep -E 'vsetvli|vsetivli|vwmacc|vfmacc|vfmul|vfcvt' | awk 'NR<=60'
} > objdump_k1_s6_seal.objdump

echo "---- [3] BYTE-EXACT IDENTITY (golden vs S6, VLEN256; cross-board correctness reconfirm) ----"
$CC -O2 -march=$MARCH -mabi=lp64d -x c verify_q4k_identity.c -c -o vdrv.o
$CC vdrv.o gold_O2.o -o verify_gold -lm
$CC vdrv.o s6_O2.o   -o verify_s6   -lm
id_ok=1
for mode in int norm; do
  taskset -c $CORE ./verify_gold $mode $K $NR $NC 0xC0FFEE og_$mode >/dev/null
  taskset -c $CORE ./verify_s6   $mode $K $NR $NC 0xC0FFEE os_$mode >/dev/null
  if cmp -s og_$mode.bin os_$mode.bin; then echo "  IDENTITY mode=$mode : IDENTICAL (0 byte mismatch) golden==S6 @VLEN256"; else echo "  IDENTITY mode=$mode : *** MISMATCH ***"; id_ok=0; fi
done
[ $id_ok -eq 1 ] && echo "  => S6 tiled kernel is byte-identical to golden ON k1/VLEN256 too" || echo "  => IDENTITY FAILED on k1"

echo "---- [4] KERNEL-AXIS PAIRED  (cold N=$N rounds; each round fresh proc; ours(S6) vs factory block-dot) ----"
echo "loadavg_mid=$(cat /proc/loadavg)"
echo "COLUMNS: fmt ours_gmacs opp_gmacs ratio_ours_over_opp"
for fmt in q4_K q5_K; do
  for i in $(seq 1 $N); do
    line=$(LD_LIBRARY_PATH="$GGML_BIN" taskset -c $CORE ./kqgemm $fmt $K $NR $NC $ITERS 0xC0FFEE)
    og=$(echo "$line" | grep -oE 'ours_gmacs=[0-9.]+' | cut -d= -f2)
    op=$(echo "$line" | grep -oE 'opp_gmacs=[0-9.]+'  | cut -d= -f2)
    rt=$(echo "$line" | grep -oE 'ratio_ours_over_opp=[0-9.]+' | cut -d= -f2)
    printf 'ROUND %-3s %s ours=%s opp=%s ratio=%s\n' "$i" "$fmt" "$og" "$op" "$rt"
  done
done
echo "loadavg_end=$(cat /proc/loadavg)"
echo "==== DONE (board scratch under $(pwd); main tree + build untouched; governor left as-found) ===="
