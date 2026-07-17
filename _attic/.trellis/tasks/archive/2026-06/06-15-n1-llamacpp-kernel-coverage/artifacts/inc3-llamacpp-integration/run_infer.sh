#!/usr/bin/env bash
# run_infer.sh — INC-3: run the FIXED greedy prompt against a given llama-cli binary,
# capture full stdout+stderr deterministically. Run ON THE BOARD.
#
#   $1 : path to llama-cli binary
#   $2 : output basename (writes <base>.stdout / <base>.stderr)
#
# Fixed experiment params: greedy (temp 0), seed 1, n=48, -no-cnv, 16 threads, same prompt.
set -euo pipefail
BIN="${1:?usage: run_infer.sh <llama-cli> <out-base>}"
OUT="${2:?usage: run_infer.sh <llama-cli> <out-base>}"

PROMPT="The capital of France is"

"$BIN" -m "$HOME/llama-2-7b-chat.Q4_0.gguf" \
    -p "$PROMPT" -n 48 --temp 0 --seed 1 -no-cnv -t 16 \
    > "${OUT}.stdout" 2> "${OUT}.stderr"
echo "BOARD_EXIT=$?"
echo "===== ${OUT}.stdout ====="
cat "${OUT}.stdout"
echo "===== TCRV counter line (if any) ====="
grep "TCRV_Q4_INTEG" "${OUT}.stderr" || echo "(no TCRV counter line)"
