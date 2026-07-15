#!/usr/bin/env bash
# robust per-function instr density: track current REAL symbol (ignore .L internal labels)
VLTAG="$1"   # vl128 or vl256
cd /tmp/a2_iqtq
objdump -d factory.o 2>/dev/null > /tmp/fdis.txt
printf "%-9s %8s %8s %8s %7s\n" fmt opp_ins opp_rvv opp_gather vsetvl
for f in iq3_s iq2_s iq2_xs iq2_xxs iq3_xxs iq4_xs tq2_0 tq1_0; do
  awk -v tgt="ggml_vec_dot_${f}_q8_K_${VLTAG}" '
    /^[0-9a-f]+ <.*>:/ {
      s=$0; sub(/^[0-9a-f]+ </,"",s); sub(/>:.*/,"",s);
      if (s !~ /^\.L/) cur=s;   # only real symbols set cur; .L labels keep cur
      next
    }
    cur==tgt && /\t/ {
      ins++;
      # opcode is the field after the raw-bytes; grab mnemonic
      if ($0 ~ /\tv[a-z]/) rvv++;
      if ($0 ~ /vlux|vlox|vluxei|vloxei/) gath++;
      if ($0 ~ /vsetvli|vsetivli/) vs++;
    }
    END{ printf "%-9s %8d %8d %8d %7d\n", tf, ins, rvv, gath, vs }
  ' tf="$f" /tmp/fdis.txt
done
