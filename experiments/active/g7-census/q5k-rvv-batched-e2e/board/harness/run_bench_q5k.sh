#!/usr/bin/env bash
# [G7 §3] q5_K@rvv batched-e2e A/B: OFF (stock RVV vl=8 vec_dot) vs ON (our weft block-dot),
# ISOLATED to the q5_K kernel (only quants.c differs). llama-batched-bench, M=npl in {1,4,8}.
# In-place atomic .so swap (impl.so uses DT_RPATH -> LD_LIBRARY_PATH cannot override). Restores
# original live .so at end. Cold-ish, load-gated, pinned core8-15, co-tenant vLLM on 0,1. NO git.
set -uo pipefail
source /opt/tcrv-toolchains/env.sh 2>/dev/null
TREE=/home/ubuntu/tcrv-llamacpp
BIN=$TREE/build-gcc15-rv64gcv/bin
SO=libggml-cpu.so.0.15.1
MODEL=${MODEL:-/home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf}
CORES=${CORES:-8-15}; TH=${TH:-8}
NPP=${NPP:-256}; NTG=${NTG:-64}; NPL=${NPL:-1,4,8}; CTX=${CTX:-4096}
ROUNDS=${ROUNDS:-10}
BB="$BIN/llama-batched-bench"
CLI="$BIN/llama-cli"
log(){ echo "[bench] $*"; }

for f in "$BIN/$SO.G7q5kOFF" "$BIN/$SO.G7q5kON" "$BB" "$CLI" "$MODEL"; do
  [ -e "$f" ] || { log "FATAL missing $f"; exit 10; }
done
# preserve current live .so (should equal G7orig / stock) for end-restore
cp -p "$BIN/$SO" "$BIN/$SO.benchsave"
BENCHSAVE_MD5=$(md5sum "$BIN/$SO.benchsave" | cut -d' ' -f1)
log "live .so saved md5=$BENCHSAVE_MD5"
log "OFF md5=$(md5sum "$BIN/$SO.G7q5kOFF"|cut -d' ' -f1)  ON md5=$(md5sum "$BIN/$SO.G7q5kON"|cut -d' ' -f1)"

swap(){ cp -pf "$BIN/$SO.G7q5k$1" "$BIN/$SO.swaptmp"; mv -f "$BIN/$SO.swaptmp" "$BIN/$SO"; }  # atomic rename
restore_live(){ cp -pf "$BIN/$SO.benchsave" "$BIN/$SO.swaptmp"; mv -f "$BIN/$SO.swaptmp" "$BIN/$SO"; }
trap 'restore_live; log "trap: live .so restored"' EXIT

echo "======================================================================"
log "board=$(uname -srm)  gcc=$($TCRV_GCC/bin/g++ --version|head -1)"
log "model=$(basename $MODEL)  cores=$CORES th=$TH  grid npp=$NPP ntg=$NTG npl=$NPL ctx=$CTX rounds=$ROUNDS"
log "loadavg_begin=$(cat /proc/loadavg)"
VLENB=$(cat /proc/cpuinfo 2>/dev/null | grep -m1 -i vlen || echo 'vlen NA')

# ---------- (A) CORRECTNESS GATE + BANNER (bit-exact => identical greedy tokens) ----------
echo "======================================================================"
log "=== CORRECTNESS GATE (greedy, fixed seed; ON must print WEFT banner, OFF must not) ==="
PROMPT="The capital of France is"
run_cli(){ # $1=OFF|ON  -> prints generated text to stdout, banner to stderr(captured)
  swap "$1"
  taskset -c "$CORES" "$CLI" -m "$MODEL" -p "$PROMPT" -n 8 -t "$TH" \
    --seed 1234 --temp 0 --top-k 1 -c 2048 --no-warmup 2> "/tmp/g7_q5k_cli_$1.err" \
    | tr -d '\r'
}
OFF_OUT=$(run_cli OFF); ON_OUT=$(run_cli ON)
echo "--- OFF banner grep ---"; grep -c "TCRV G7-S3 WEFT" "/tmp/g7_q5k_cli_OFF.err" || true
echo "--- ON  banner grep ---"; grep -m1 "TCRV G7-S3 WEFT" "/tmp/g7_q5k_cli_ON.err" || echo "NO-BANNER(ON) *** UNEXPECTED ***"
OFF_H=$(printf '%s' "$OFF_OUT" | md5sum | cut -d' ' -f1)
ON_H=$(printf '%s' "$ON_OUT" | md5sum | cut -d' ' -f1)
log "OFF output md5=$OFF_H"
log "ON  output md5=$ON_H"
log "GREEDY_IDENTICAL=$([ "$OFF_H" = "$ON_H" ] && echo YES || echo NO)"
echo "--- OFF tail ---"; printf '%s\n' "$OFF_OUT" | tail -3
echo "--- ON  tail ---"; printf '%s\n' "$ON_OUT" | tail -3

# ---------- (B) BATCHED-BENCH A/B (interleaved OFF/ON per round) ----------
echo "======================================================================"
log "=== BATCHED-BENCH A/B (N=$ROUNDS interleaved rounds) ==="
bench(){ # $1=OFF|ON $2=round -> emits jsonl lines tagged
  swap "$1"
  taskset -c "$CORES" "$BB" -m "$MODEL" -c "$CTX" -b 2048 -ub 512 \
     -npp "$NPP" -ntg "$NTG" -npl "$NPL" -t "$TH" -tb "$TH" \
     --output-format jsonl 2>>"/tmp/g7_q5k_bb_$1.err" \
   | grep -E '^\{' | sed "s/^/G7BB side=$1 round=$2 /"
}
# warmup (drop) to page-cache the model
swap OFF; taskset -c "$CORES" "$BB" -m "$MODEL" -c "$CTX" -npp 64 -ntg 8 -npl 1 -t "$TH" -tb "$TH" --output-format jsonl >/dev/null 2>&1 || true
for r in $(seq 1 $ROUNDS); do
  LA=$(cat /proc/loadavg | cut -d' ' -f1)
  echo "# round=$r loadavg=$LA"
  bench OFF "$r"
  bench ON  "$r"
done
log "loadavg_end=$(cat /proc/loadavg)"

# cleanup + verify restore
restore_live
LIVE_NOW=$(md5sum "$BIN/$SO"|cut -d' ' -f1)
log "live .so restored md5=$LIVE_NOW (save=$BENCHSAVE_MD5) match=$([ "$LIVE_NOW" = "$BENCHSAVE_MD5" ] && echo YES || echo NO)"
rm -f "$BIN/$SO.benchsave" "$BIN/$SO.swaptmp"
pkill -u "$(id -un)" -f llama-batched-bench 2>/dev/null || true
pkill -u "$(id -un)" -f llama-cli 2>/dev/null || true
log "stray_bb=$(pgrep -c -f llama-batched-bench || echo 0) stray_cli=$(pgrep -c -f llama-cli || echo 0)"
log "=== BENCH DONE ==="
