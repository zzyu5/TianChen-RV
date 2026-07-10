#!/usr/bin/env bash
# [G3 T4b seal-fix] Contention-clean PAIRED-INTERLEAVED bench (phase-per-size).
# Variants, all SAME recipe (taskset -c 8-15, -t 8, llama-bench --no-warmup -o json):
#   A       = tcrv bin + .A-q4kON  (OUR emitted vl=8 q4_K repack GEMM/GEVM)
#   Bstock  = upstream-native bin  (real upstream ggml block-dot)
#   Bq4kOFF = tcrv bin + .B-q4kOFF  (tcrv tree, q4_K gate OFF; kernel-isolated block-dot)
# Round-level interleave A,Bstock,Bq4kOFF per round to cancel slow board drift.
# NEIGHBOR gate: before each invocation (we are idle), sample 1s total CPU busy% from
# /proc/stat (my own 8 cores are NOT running at that instant, so busy% = neighbors). ABORT
# if busy% >= CPUGATE. loadavg logged as evidence (raw loadavg is contaminated by my own
# 8 cores mid-campaign, so it's evidence not gate). EXIT trap force-restores A-tree. NO git.
# usage: board_bench_paired.sh "<size1:rounds1> <size2:rounds2> ..." [inner]
#   size: pp value e.g. 128,256,512  OR  tgNN for decode (n-gen NN)
set -u
. /opt/tcrv-toolchains/env.sh 2>/dev/null
PLAN="${1:-128:4 512:3 tg32:4 256:3}"
INNER="${2:-1}"
CPUGATE="${CPUGATE:-15}"     # percent of all 64 cores busy by non-me => neighbor contention
MODEL=/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf
TCRV=/home/ubuntu/tcrv-llamacpp/build-gcc15-rv64gcv/bin
UP=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
OUTDIR=/tmp/tf_paired
STOCK_MD5=75f20b5fddf2faf2b18d7a88b1bfb4b5
GSYM=tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K
mkdir -p "$OUTDIR"; rm -f "$OUTDIR"/ABORT 2>/dev/null

restore_atree() {
  cp -f "$TCRV/libggml-cpu.so.0.15.1.B-q4kOFF" "$TCRV/libggml-cpu.so.0.15.1"
  local m g; m=$(md5sum "$TCRV/libggml-cpu.so.0.15.1" | awk '{print $1}')
  g=$(nm -C "$TCRV/libggml-cpu.so.0.15.1" 2>/dev/null | grep -c "$GSYM")
  echo "[EXIT-RESTORE] live .so md5=$m (expect $STOCK_MD5)  gemm_sym=$g (expect 0)"
}
trap restore_atree EXIT

lg1() { awk '{print $1}' /proc/loadavg; }
# 1-second total CPU busy% across all cores (instantaneous neighbor signal while we're idle)
cpubusy() {
  read -r _ a b c d e f g h _ < /proc/stat
  local idle0=$((d+e)) tot0=$((a+b+c+d+e+f+g+h))
  sleep 1
  read -r _ a b c d e f g h _ < /proc/stat
  local idle1=$((d+e)) tot1=$((a+b+c+d+e+f+g+h))
  local dt=$((tot1-tot0)) di=$((idle1-idle0))
  [ "$dt" -le 0 ] && { echo 0; return; }
  echo $(( (100*(dt-di)) / dt ))
}

# run one variant, one size spec, one round -> JSON (returns 1 to signal ABORT)
run_one() {
  local VAR="$1" SIZE="$2" RND="$3"
  local BUSY LOAD; BUSY=$(cpubusy); LOAD=$(lg1)
  if [ "$BUSY" -ge "$CPUGATE" ]; then
    echo "  [GATE-ABORT] non-me CPU busy=${BUSY}% (>= ${CPUGATE}%) loadavg1=$LOAD -- honest abort, no fabricated samples"
    touch "$OUTDIR/ABORT"; return 1
  fi
  local BIN LD PPARG NARG TAG
  if [[ "$SIZE" == tg* ]]; then PPARG=0; NARG="${SIZE#tg}"; TAG="$SIZE"; else PPARG="$SIZE"; NARG=0; TAG="pp${SIZE}"; fi
  if [ "$VAR" = "Bstock" ]; then BIN="$UP/llama-bench"; LD="$UP"
  else
    if [ "$VAR" = "A" ]; then cp -f "$TCRV/libggml-cpu.so.0.15.1.A-q4kON" "$TCRV/libggml-cpu.so.0.15.1"
    else                     cp -f "$TCRV/libggml-cpu.so.0.15.1.B-q4kOFF" "$TCRV/libggml-cpu.so.0.15.1"; fi
    BIN="$TCRV/llama-bench"; LD="$TCRV"
  fi
  export LD_LIBRARY_PATH="$LD:${LD_LIBRARY_PATH:-}"
  local OUT="$OUTDIR/${VAR}_${TAG}_r${RND}.json" ERR="$OUTDIR/${VAR}_${TAG}_r${RND}.err"
  local T0=$SECONDS
  taskset -c 8-15 "$BIN" -m "$MODEL" -p "$PPARG" -n "$NARG" -t 8 -r "$INNER" --no-warmup -o json > "$OUT" 2> "$ERR"
  local EC=$? DT=$((SECONDS-T0))
  local BAN AVG; BAN=$(grep -cE "TCRV EMITTED GE[MV][MV]\(q4_K_16x1 VLEN128" "$ERR" 2>/dev/null)
  AVG=$(grep -oE '"avg_ts": *[0-9.]+' "$OUT" 2>/dev/null | grep -oE '[0-9.]+' | paste -sd, -)
  echo "  [$VAR $TAG r$RND] exit=$EC ${DT}s pre{busy=${BUSY}%,load=$LOAD} banner=$BAN avg_ts=[$AVG]"
}

echo "### PAIRED bench PLAN='$PLAN' inner=$INNER CPUGATE=${CPUGATE}% start_load=$(cat /proc/loadavg)"
for phase in $PLAN; do
  SIZE="${phase%%:*}"; RNDS="${phase##*:}"
  echo "== PHASE size=$SIZE rounds=$RNDS =="
  for r in $(seq 1 "$RNDS"); do
    echo " -- $SIZE round $r --"
    for VAR in A Bstock Bq4kOFF; do
      run_one "$VAR" "$SIZE" "$r" || { echo "### ABORTED during $SIZE r$r $VAR"; exit 0; }
    done
  done
  echo "  [phase $SIZE done] load=$(cat /proc/loadavg)"
done
echo "### DONE all phases. JSON count: $(ls "$OUTDIR"/*.json 2>/dev/null | wc -l)"
