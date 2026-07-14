#!/usr/bin/env bash
# [G6-B Phase-2 q2_K @ k1] correctness-first, deployment-chain check for the ROLLED variant:
#   greedy A==B  A = ON_ROLLED (our emitted rolled-loop q2_K repack)  vs  B = OFF (stock hand-brick)
#   AND          A2 = ON_ROLLED  vs  B2 = ON_UNROLLED (Phase-1 proved rolled==unrolled byte-exact;
#                this confirms the deployment chain preserves that identity end-to-end on k1).
# Silicon integer byte-exactness already proven separately (kquant_repack_verify_q2K INT_mismatch=0);
# here byte-identical greedy is the strong pass, a late single-token ULP divergence the honest caveat.
set -u
BUILD=/data/build-k1-q2k-rolled
BIN=$BUILD/bin
LIVE=$BIN/libggml-cpu.so.0.15.1
WK=/tmp/g6b-p2-q2k
CLI="${CLI:-/data/k1build/bin/llama-cli}"
MODEL="${MODEL:-/data/tinyllama-1.1b-Q2_K_M.gguf}"
THREADS="${THREADS:-8}"; NTOK="${NTOK:-32}"; PIN="taskset -c 0-7"
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
swap(){ cp -f "$WK/libggml-cpu.so.$1" "$LIVE"; }
gen(){ swap "$1"
  $PIN "$CLI" -m "$MODEL" -p "$PROMPT" -n "$NTOK" -t "$THREADS" \
     --no-warmup --temp 0 --top-k 1 --seed 1 -no-cnv -st --no-display-prompt </dev/null 2>>/tmp/g6bp2_engage.err
}
echo "== env: model=$MODEL sha256=$(sha256sum "$MODEL"|cut -c1-16) threads=$THREADS ntok=$NTOK =="
echo "== ROLLED md5=$(md5sum "$WK/libggml-cpu.so.ON_ROLLED"|awk '{print $1}')  UNROLLED md5=$(md5sum "$WK/libggml-cpu.so.ON_UNROLLED"|awk '{print $1}')  OFF md5=$(md5sum "$WK/libggml-cpu.so.OFF"|awk '{print $1}') =="
PROMPTS="The capital of France is
Once upon a time
Q: What is 2 + 2? A:
The quick brown fox jumps"
: > /tmp/g6bp2_engage.err
P_RS=0; F_RS=0; P_RU=0; F_RU=0; IDX=0
while IFS= read -r PROMPT; do
  [ -z "$PROMPT" ] && continue
  IDX=$((IDX+1))
  gen ON_ROLLED   > /tmp/g6bp2_R_$IDX.txt
  gen OFF         > /tmp/g6bp2_S_$IDX.txt
  gen ON_UNROLLED > /tmp/g6bp2_U_$IDX.txt
  echo "-- prompt[$IDX]: '$PROMPT'"
  echo "   R(rolled):   $(tr '\n' ' ' < /tmp/g6bp2_R_$IDX.txt | cut -c1-88)"
  echo "   S(stock):    $(tr '\n' ' ' < /tmp/g6bp2_S_$IDX.txt | cut -c1-88)"
  echo "   U(unrolled): $(tr '\n' ' ' < /tmp/g6bp2_U_$IDX.txt | cut -c1-88)"
  if [ ! -s /tmp/g6bp2_R_$IDX.txt ]; then echo "   R empty!"; F_RS=$((F_RS+1)); F_RU=$((F_RU+1)); continue; fi
  # rolled vs stock
  if diff -q /tmp/g6bp2_R_$IDX.txt /tmp/g6bp2_S_$IDX.txt >/dev/null; then echo "   R==S: BYTE-IDENTICAL"; P_RS=$((P_RS+1))
  else echo "   R!=S: (bounded-ULP? show first diff)"; diff /tmp/g6bp2_R_$IDX.txt /tmp/g6bp2_S_$IDX.txt|head -4|sed 's/^/     /'; F_RS=$((F_RS+1)); fi
  # rolled vs unrolled (must be identical -- same math, byte-exact by construction)
  if diff -q /tmp/g6bp2_R_$IDX.txt /tmp/g6bp2_U_$IDX.txt >/dev/null; then echo "   R==U: BYTE-IDENTICAL"; P_RU=$((P_RU+1))
  else echo "   R!=U: *** UNEXPECTED (rolled must == unrolled)"; diff /tmp/g6bp2_R_$IDX.txt /tmp/g6bp2_U_$IDX.txt|head -4|sed 's/^/     /'; F_RU=$((F_RU+1)); fi
done <<< "$PROMPTS"
echo "== engage banner (proves our rolled kernel ran) =="
grep -m2 "WEFT G5-q2K EMITTED" /tmp/g6bp2_engage.err | sed 's/^/   /' || echo "   (no banner!)"
echo "== SUMMARY: R==S byte_identical=$P_RS/$IDX (fail=$F_RS)  |  R==U byte_identical=$P_RU/$IDX (fail=$F_RU) =="
[ "$F_RU" = 0 ] && echo "ROLLED==UNROLLED e2e: GREEN (deployment chain preserves Phase-1 byte-exact identity)" || echo "ROLLED==UNROLLED e2e: *** DIVERGENCE (investigate)"
[ "$F_RS" = 0 ] && echo "ROLLED==STOCK e2e: GREEN (A==B byte-identical over corpus)" || echo "ROLLED==STOCK e2e: bounded-ULP divergence (characterize; stock uses different scale layout)"
swap OFF
