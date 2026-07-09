#!/usr/bin/env bash
# run_k1_identity_timing.sh — [SEL-1 T4a] resume: sections [3] IDENTITY + [4] TIMING only.
# Reuses the .o already compiled by run_k1_kquant_t4a.sh (gold_O2.o s6_O2.o kq5k_O2.o kqdrv.o kqgemm),
# so NO recompile. Closes gate ⑤ (k1 kernel-axis paired) + reconfirms byte-exact on k1/VLEN256.
set -uo pipefail
K="${K:-2048}"; NR="${NR:-64}"; NC="${NC:-512}"; ITERS="${ITERS:-20}"; N="${N:-12}"; CORE="${CORE:-7}"
MARCH="${MARCH:-rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs}"; CC="${CC:-clang}"; GGML_BIN="${GGML_BIN:-/data/k1build-stock/bin}"
echo "==== T4a k1 RESUME identity+timing  K=$K nr=$NR nc=$NC iters=$ITERS N=$N core=$CORE ===="
for f in gold_O2.o s6_O2.o kq5k_O2.o kqdrv.o kqgemm; do [ -e "$f" ] || { echo "MISSING $f"; exit 40; }; done
echo "textB(.text bytes): $(for o in gold_O2.o s6_O2.o kq5k_O2.o; do h=$(objdump -h $o|awk '$2==".text"{print $3}'); printf '%s=%d ' $o $((16#$h)); done)"

echo "---- [3] BYTE-EXACT IDENTITY (golden vs S6, VLEN256) ----"
$CC -O2 -march=$MARCH -mabi=lp64d -x c verify_q4k_identity.c -c -o vdrv.o
$CC vdrv.o gold_O2.o -o verify_gold -lm
$CC vdrv.o s6_O2.o   -o verify_s6   -lm
id_ok=1
for mode in int norm; do
  taskset -c $CORE ./verify_gold $mode $K $NR $NC 0xC0FFEE og_$mode >/dev/null
  taskset -c $CORE ./verify_s6   $mode $K $NR $NC 0xC0FFEE os_$mode >/dev/null
  if cmp -s og_$mode.bin os_$mode.bin; then echo "  IDENTITY mode=$mode : IDENTICAL (0 byte mismatch) golden==S6 @VLEN256"; else echo "  IDENTITY mode=$mode : *** MISMATCH ***"; id_ok=0; fi
done
[ $id_ok -eq 1 ] && echo "  => S6 tiled kernel byte-identical to golden ON k1/VLEN256" || echo "  => IDENTITY FAILED on k1"

echo "---- [4] KERNEL-AXIS PAIRED (cold N=$N; each round fresh proc; ours(S6) vs factory block-dot) ----"
echo "loadavg_begin=$(cat /proc/loadavg)"
echo "COLUMNS: ROUND i fmt ours_gmacs opp_gmacs ratio_ours_over_opp"
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
echo "==== DONE-RESUME ===="
