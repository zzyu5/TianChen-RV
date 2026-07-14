#!/usr/bin/env bash
# [G7 §3 v2] q5_K@rvv batched-e2e A/B — bench-only (llama-cli decode does not thread on this board;
# batched-bench does). Engagement proof = WEFT banner in the ON batched-bench stderr (fires on the
# first q5_K vec_dot in the ACTUAL measured path). Bit-exactness inherited from M-sweep VERIFY_FAIL=0.
# OFF=stock RVV vl=8 vec_dot / ON=our weft block-dot; only quants.c differs. Atomic in-place .so swap
# (impl.so DT_RPATH). Interleaved N rounds, core8-15 pinned, load-gated, restores live .so. NO git.
set -uo pipefail
source /opt/tcrv-toolchains/env.sh 2>/dev/null
TREE=/home/ubuntu/tcrv-llamacpp
BIN=$TREE/build-gcc15-rv64gcv/bin
SO=libggml-cpu.so.0.15.1
MODEL=${MODEL:-/home/ubuntu/models/tinyllama-q5_K_M.gguf}
CORES=${CORES:-8-15}; TH=${TH:-8}
NPP=${NPP:-32}; NTG=${NTG:-24}; NPL=${NPL:-1,4,8}; CTX=${CTX:-2048}
ROUNDS=${ROUNDS:-10}
BB="$BIN/llama-batched-bench"
log(){ echo "[bench] $*"; }
for f in "$BIN/$SO.G7q5kOFF" "$BIN/$SO.G7q5kON" "$BB" "$MODEL"; do [ -e "$f" ] || { log "FATAL missing $f"; exit 10; }; done

cp -p "$BIN/$SO" "$BIN/$SO.benchsave"; BSAVE=$(md5sum "$BIN/$SO.benchsave"|cut -d' ' -f1)
log "live .so saved md5=$BSAVE   OFF=$(md5sum "$BIN/$SO.G7q5kOFF"|cut -d' ' -f1) ON=$(md5sum "$BIN/$SO.G7q5kON"|cut -d' ' -f1)"
swap(){ cp -pf "$BIN/$SO.G7q5k$1" "$BIN/$SO.swaptmp"; mv -f "$BIN/$SO.swaptmp" "$BIN/$SO"; }
restore_live(){ cp -pf "$BIN/$SO.benchsave" "$BIN/$SO.swaptmp"; mv -f "$BIN/$SO.swaptmp" "$BIN/$SO"; }
trap 'restore_live; log "trap: live .so restored"' EXIT

log "board=$(uname -srm) gcc=$($TCRV_GCC/bin/g++ --version|head -1)"
log "model=$(basename $MODEL) cores=$CORES th=$TH grid npp=$NPP ntg=$NTG npl=$NPL ctx=$CTX rounds=$ROUNDS"
log "loadavg_begin=$(cat /proc/loadavg)"

: > /tmp/g7_q5k_bb_OFF.err; : > /tmp/g7_q5k_bb_ON.err
bench(){ # $1=OFF|ON $2=round
  swap "$1"
  taskset -c "$CORES" "$BB" -m "$MODEL" -c "$CTX" -b 2048 -ub 512 \
     -npp "$NPP" -ntg "$NTG" -npl "$NPL" -t "$TH" -tb "$TH" --output-format jsonl \
     2>>"/tmp/g7_q5k_bb_$1.err" | grep -E '^\{' | sed "s/^/G7BB side=$1 round=$2 /"
}
# warmup (page-cache the model; dropped)
swap OFF; taskset -c "$CORES" "$BB" -m "$MODEL" -c "$CTX" -npp 16 -ntg 4 -npl 1 -t "$TH" -tb "$TH" --output-format jsonl >/dev/null 2>&1 || true
log "=== BATCHED-BENCH A/B (N=$ROUNDS interleaved) ==="
for r in $(seq 1 $ROUNDS); do
  echo "# round=$r loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  bench OFF "$r"
  bench ON  "$r"
done
log "loadavg_end=$(cat /proc/loadavg)"

# engagement proof: WEFT banner must appear in ON bench stderr, and NOT in OFF
log "=== ENGAGEMENT (WEFT banner) ==="
echo "ON  WEFT banner count: $(grep -c 'TCRV G7-S3 WEFT q5_K' /tmp/g7_q5k_bb_ON.err 2>/dev/null || echo 0)"
echo "OFF WEFT banner count: $(grep -c 'TCRV G7-S3 WEFT q5_K' /tmp/g7_q5k_bb_OFF.err 2>/dev/null || echo 0)"
grep -m1 'TCRV G7-S3 WEFT q5_K' /tmp/g7_q5k_bb_ON.err 2>/dev/null || echo "(no ON banner line)"

restore_live
LIVE=$(md5sum "$BIN/$SO"|cut -d' ' -f1)
log "live .so restored md5=$LIVE (save=$BSAVE) match=$([ "$LIVE" = "$BSAVE" ] && echo YES || echo NO)"
rm -f "$BIN/$SO.benchsave" "$BIN/$SO.swaptmp"
pkill -u "$(id -un)" -f llama-batched-bench 2>/dev/null || true
log "stray_bb=$(pgrep -c -f llama-batched-bench || echo 0)"
log "=== BENCH DONE ==="
