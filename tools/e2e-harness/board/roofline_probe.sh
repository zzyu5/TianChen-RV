#!/usr/bin/env bash
# roofline_probe.sh -- board-side roofline calibration: the two physical ceilings
# every decode/prefill parity judgment is measured against.
#   (A) sustained READ bandwidth  (membw_probe.c) : single-core + N-core aggregate
#       on the SAME core set the e2e decode uses (shared DRAM channel ceiling).
#   (B) peak int8 compute (int8_peak_probe.c)     : register-resident vwmacc GMACs/GOPS.
# Emits one machine-parseable ROOFLINE block. Compiles the probes on-board with the
# board compiler so the ceiling reflects the board's own toolchain/uarch.
#
# Env: CORES (e.g. "0-3" or "8-11")  MARCH (default rv64gcv)  CC (default clang||gcc)
#      MEMBW_MIB (default 256)  MEMBW_ITERS (default 40)  PEAK_ITERS (default 400000000)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORES="${CORES:-8-11}"
MARCH="${MARCH:-rv64gcv}"
MEMBW_MIB="${MEMBW_MIB:-256}"
MEMBW_ITERS="${MEMBW_ITERS:-40}"
PEAK_ITERS="${PEAK_ITERS:-400000000}"
CC="${CC:-}"
if [ -z "$CC" ]; then
  for c in clang-18 clang-17 clang gcc; do command -v "$c" >/dev/null 2>&1 && { CC="$c"; break; }; done
fi
[ -n "$CC" ] || { echo "ROOFLINE_FATAL no compiler"; exit 2; }

# core list -> array (expand "a-b")
lo="${CORES%%-*}"; hi="${CORES##*-}"; [ "$hi" = "$CORES" ] && hi="$lo"
CORELIST=(); for c in $(seq "$lo" "$hi"); do CORELIST+=("$c"); done
NCORE=${#CORELIST[@]}

echo "== ROOFLINE PROBE =="
echo "compiler=$($CC --version 2>/dev/null | head -1)"
echo "march=$MARCH  cores=$CORES ($NCORE cores)  membw_mib=$MEMBW_MIB iters=$MEMBW_ITERS  peak_iters=$PEAK_ITERS"
echo "governor=$(cat /sys/devices/system/cpu/cpu${lo}/cpufreq/scaling_governor 2>/dev/null || echo NA)"
echo "cur_khz=$(cat /sys/devices/system/cpu/cpu${lo}/cpufreq/scaling_cur_freq 2>/dev/null || echo NA)  max_khz=$(cat /sys/devices/system/cpu/cpu${lo}/cpufreq/cpuinfo_max_freq 2>/dev/null || echo NA)"

BW="$HERE/membw_probe"; PK="$HERE/int8_peak_probe"
echo "== compile =="
$CC -O3 -march="$MARCH" -o "$BW" "$HERE/membw_probe.c" 2>&1 | sed 's/^/[membw] /' || { echo "ROOFLINE_FATAL membw compile"; exit 3; }
$CC -O2 -march="$MARCH" -o "$PK" "$HERE/int8_peak_probe.c" 2>&1 | sed 's/^/[peak] /'  || { echo "ROOFLINE_FATAL peak compile"; exit 3; }

# self-check: the hot loop must retain vwmacc (no closed-form strength-reduce)
NVW=$(objdump -d "$PK" 2>/dev/null | grep -c 'vwmacc')
echo "SELFCHECK int8_peak vwmacc_in_binary=$NVW (expect >=8 => not folded)"

echo "== (A) READ BANDWIDTH =="
# single core
S1=$(taskset -c "${CORELIST[0]}" "$BW" "$MEMBW_MIB" "$MEMBW_ITERS")
echo "single_core[core ${CORELIST[0]}]: $S1"
G1=$(echo "$S1" | sed -nE 's/.*MEMBW_GBs=([0-9.]+).*/\1/p')

# N-core aggregate: launch one pinned copy per core concurrently, sum GB/s
echo "aggregate[$NCORE cores concurrent]:"
pids=(); tmpd="$(mktemp -d)"
for c in "${CORELIST[@]}"; do
  ( taskset -c "$c" "$BW" "$MEMBW_MIB" "$MEMBW_ITERS" > "$tmpd/c$c.out" 2>&1 ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
AGG=0
for c in "${CORELIST[@]}"; do
  line="$(cat "$tmpd/c$c.out")"
  g=$(echo "$line" | sed -nE 's/.*MEMBW_GBs=([0-9.]+).*/\1/p')
  echo "  core $c: $line"
  AGG=$(awk -v a="$AGG" -v g="$g" 'BEGIN{printf "%.3f", a+g}')
done
rm -rf "$tmpd"
echo "AGG_READ_GBs=$AGG  SINGLE_READ_GBs=$G1"

echo "== (B) PEAK INT8 COMPUTE =="
P1=$(taskset -c "${CORELIST[0]}" "$PK" "$PEAK_ITERS")
echo "single_core[core ${CORELIST[0]}]: $P1"
PGMAC=$(echo "$P1" | sed -nE 's/.*GMACs=([0-9.]+).*/\1/p')
PGOPS=$(echo "$P1" | sed -nE 's/.*GOPS=([0-9.]+).*/\1/p')

echo "== ROOFLINE SUMMARY =="
echo "ROOFLINE read_single_GBs=$G1 read_agg${NCORE}c_GBs=$AGG int8_peak_1c_GMACs=$PGMAC int8_peak_1c_GOPS=$PGOPS cores=$CORES march=$MARCH cc=$CC vwmacc_check=$NVW"
rm -f "$BW" "$PK"
