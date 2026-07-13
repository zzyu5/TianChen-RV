#!/usr/bin/env bash
# Launcher: scp the M5 q4_0 matmul-internal patch + board runner onto k1, start under nohup
# (survives ssh disconnect; poll /tmp/g6m5q40/run.log for ALL_DONE_q40_m5).
set -uo pipefail
BOARD="${BOARD:-k1}"
BD="/tmp/g6m5q40"
H="tools/e2e-harness/board/g6-m5-ime-matmul-internal"
ssh "$BOARD" "mkdir -p $BD"
scp "$H/forward-route-patch-q40-matmul.py" "$BOARD:$BD/" >/dev/null
scp "$H/run-m5-q40-board.sh" "$BOARD:$BD/" >/dev/null
ssh "$BOARD" "cd $BD && rm -f run.log && PP='${PP:-32}' REPS='${REPS:-4}' PASSES='${PASSES:-3}' THREADS='${THREADS:-4}' nohup bash run-m5-q40-board.sh > run.log 2>&1 & echo launched pid=\$!"
echo "launched q40 M5; poll $BD/run.log for ALL_DONE_q40_m5"
