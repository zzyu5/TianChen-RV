#!/usr/bin/env bash
# probe_fix_rvv.sh — CORRECTED opponent/ours instruction-class probe (read-only, no measurement).
# WHY: the inline probe in run_rvv_dequant_p1.sh used grep -E '\tv[a-z]'; GNU grep -E does NOT
# interpret \t (warns "stray \ before t") => the pattern silently degraded to literal 'tv[a-z]'.
# All ins/rvv/gather counts from that probe are INVALID. This re-probes with tab-field-aware awk:
# objdump -d lines are TAB separated: addr \t hexbytes \t mnemonic \t operands. Classify on $3.
# Dual method: (m1) objdump --disassemble=<sym>  (m2) whole-file body extraction. Disagree => VOID-PROBE.
set -uo pipefail
GGML=/home/ubuntu/llama.cpp-upstream-native/build-clang18-rv64gcv/bin
RDIR=/tmp/g8_p1_dq_rvv
OUT=/tmp/g8_p1_probe_fixed.txt

ssh rvv "set -uo pipefail; source /opt/tcrv-toolchains/env.sh; cd $RDIR
: > $OUT
# classify(file) -> ins rvv gather vset  ; mnemonic = 3rd TAB field
classify(){ awk -F'\t' '
  \$3 ~ /^[a-z]/ { ins++
                   if (\$3 ~ /^v/) rvv++
                   if (\$3 ~ /^(vlux|vloxei|vrgather)/) gat++
                   if (\$3 ~ /^vset/) vs++ }
  END { printf \"ins=%d rvv=%d gather=%d vset=%d\", ins+0, rvv+0, gat+0, vs+0 }' \"\$1\"; }

echo '=== OPPONENT (as-shipped libggml-base.so) ===' | tee -a $OUT
echo \"# base_md5=\$(md5sum $GGML/libggml-base.so | cut -d' ' -f1)\" | tee -a $OUT
objdump -d $GGML/libggml-base.so > lib_base.txt 2>/dev/null
for f in iq2_xs iq2_s nvfp4; do
  sym=dequantize_row_\$f
  objdump --disassemble=\$sym $GGML/libggml-base.so > p_m1_\$f.txt 2>/dev/null
  awk -v s=\"<\${sym}>:\" 'g&&/^[0-9a-f]+ </{exit} \$0 ~ s{g=1} g' lib_base.txt > p_m2_\$f.txt
  A=\$(classify p_m1_\$f.txt); B=\$(classify p_m2_\$f.txt)
  ADDR=\$(grep -oE '^[0-9a-f]+ <'\$sym'>:' p_m1_\$f.txt | cut -d' ' -f1)
  AG=DUAL-AGREE; [ \"\$A\" != \"\$B\" ] && AG=DUAL-DISAGREE-VOID-PROBE
  echo \"OPP \$sym @\$ADDR  m1[\$A]  m2[\$B]  => \$AG\" | tee -a $OUT
done

echo '=== OURS (weft-emitted dequant kernels, .o built this session) ===' | tee -a $OUT
for f in iq2_xs iq2_s nvfp4; do
  objdump -d kd_\$f.o > p_ours_\$f.txt 2>/dev/null
  C=\$(classify p_ours_\$f.txt)
  SP=\$(awk -F'\t' '\$3 ~ /^(sd|ld|sw|lw)\$/ && \$4 ~ /\(sp\)/ {n++} END{print n+0}' p_ours_\$f.txt)
  echo \"OURS dequant_\$f  [\$C] true_sp_spill=\$SP size=\$(wc -c < kd_\$f.o)B\" | tee -a $OUT
done
cat $OUT
"
