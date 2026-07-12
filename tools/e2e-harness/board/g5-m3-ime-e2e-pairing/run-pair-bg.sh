#!/usr/bin/env bash
# Launcher: scp the paired-bench harness + format bridge patch onto k1 and start
# run_pair_board.sh under nohup (survives ssh disconnect; poll <BD>/run.log for
# ALL_DONE_<fmt>). ONE format per call:
#   ./run-pair-bg.sh <q4_0|q8_0|q4_K>
set -uo pipefail
FMT="${1:?usage: run-pair-bg.sh <q4_0|q8_0|q4_K>}"
BOARD="${BOARD:-k1}"
PP_LIST="${PP_LIST:-32 256}"; TG="${TG:-64}"; REPS="${REPS:-5}"; PASSES="${PASSES:-2}"
BD="/tmp/g5m3pair/$FMT"
H="tools/e2e-harness/board/g5-m3-ime-e2e-pairing"

case "$FMT" in
  q4_0) PATCH="tools/e2e-harness/board/g5-m3-ime-q4_0/forward-route-patch.py"
        MODEL="/home/bianbu/tcrv-k1-llama/models/tinyllama-q4_0.gguf"; ENVV="TCRV_IME_Q40_BRIDGE"; MARK="TCRV-IME-Q40-BRIDGE" ;;
  q8_0) PATCH="tools/e2e-harness/board/g5-m3-ime-q8_0/forward-route-patch-q8.py"
        MODEL="/data/tinyllama-q8_0.gguf"; ENVV="TCRV_IME_Q80_BRIDGE"; MARK="TCRV-IME-Q80-BRIDGE" ;;
  q4_K) PATCH="tools/e2e-harness/board/g5-m3-ime-q4_K/forward-route-patch-q4k.py"
        MODEL="/data/tinyllama-1.1b-Q4_K_M.gguf"; ENVV="TCRV_IME_Q4K_BRIDGE"; MARK="TCRV-IME-Q4K-BRIDGE" ;;
  *) echo "unknown format $FMT"; exit 2 ;;
esac
PATCH_BN="$(basename "$PATCH")"

ssh "$BOARD" "mkdir -p $BD"
scp "$PATCH" "$BOARD:$BD/$PATCH_BN" >/dev/null
scp "$H/run_pair_board.sh" "$BOARD:$BD/" >/dev/null

ssh "$BOARD" "cd $BD && rm -f run.log && nohup bash run_pair_board.sh '$FMT' '$MODEL' '$PATCH_BN' '$ENVV' '$MARK' '$PP_LIST' '$TG' '$REPS' '$PASSES' > run.log 2>&1 & echo launched pid=\$!"
echo "launched $FMT; poll $BD/run.log for ALL_DONE_$FMT"
