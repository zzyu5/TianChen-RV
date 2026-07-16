#!/usr/bin/env bash
# C1 loop-order consumption census + hollow-gate check. Self-contained; run from repo root.
set -u
cd "$(git rev-parse --show-toplevel)"
export PATH=/usr/lib/llvm-20/bin:$PATH
BIN=build-weft/bin/weft-opt
T=$(mktemp -d)
FMT="iq2-s iq2-xs iq2-xxs iq4-nl iq4-xs mxfp4 q2-K q3-K q4-0 q4-1 q4-K q5-0 q5-1 q5-K q6-K q8-0 tq1-0 tq2-0"
echo "== A. per-format loop_order consumption (stamp present? attr flip changes emitted C?) =="
Y=0; N=0
for f in $FMT; do
  S=test/Conversion/RVV/rvv-emit-quant-contraction-$f-repack-gemm-prefill-vlen128.mlir
  $BIN $S --weft-rvv-lower-quant-contraction=march=rv64gcv 2>/dev/null > $T/s.mlir
  ST=$(grep -c 'weft_rvv.loop_order = ' $T/s.mlir)
  sed 's/weft_rvv.loop_order = "[a-z_]*"/weft_rvv.loop_order = "col_outer"/; s/weft_rvv.loop_order_selection_reason = "[a-z_]*"/weft_rvv.loop_order_selection_reason = "measured"/' $T/s.mlir > $T/c.mlir
  sed 's/weft_rvv.loop_order = "[a-z_]*"/weft_rvv.loop_order = "row_outer"/; s/weft_rvv.loop_order_selection_reason = "[a-z_]*"/weft_rvv.loop_order_selection_reason = "measured"/' $T/s.mlir > $T/r.mlir
  $BIN $T/c.mlir --weft-rvv-lower-to-emitc 2>/dev/null > $T/oc.txt
  $BIN $T/r.mlir --weft-rvv-lower-to-emitc 2>/dev/null > $T/or.txt
  D=$(diff $T/oc.txt $T/or.txt | wc -l)
  if [ "$ST" -ge 1 ] && [ "$D" -gt 0 ]; then Y=$((Y+1)); V=YES; else N=$((N+1)); V="NO"; fi
  printf "  %-8s stamp=%s diff=%-4s %s\n" "$f" "$ST" "$D" "$V"
done
echo "  => CONSUMES $Y/18, NOT $N/18"
echo "== B. hollow-gate check: every col-outer fixture must be green + RED under attr flip =="
H=0
for F in test/Conversion/RVV/*col-outer*.mlir; do
  $BIN $F --weft-rvv-lower-to-emitc 2>/dev/null | FileCheck $F >/dev/null 2>&1 && P=green || P=RED
  sed 's/weft_rvv.loop_order = "col_outer"/weft_rvv.loop_order = "row_outer"/' $F > $T/n.mlir
  $BIN $T/n.mlir --weft-rvv-lower-to-emitc 2>/dev/null | FileCheck $F >/dev/null 2>&1 && NG=green || NG=RED
  [ "$P" = green ] && [ "$NG" = RED ] && R=OK || { R="HOLLOW"; H=$((H+1)); }
  printf "  %-56s pos=%-6s neg=%-5s %s\n" "$(basename $F .mlir)" "$P" "$NG" "$R"
done
echo "  => HOLLOW = $H"
rm -rf $T
