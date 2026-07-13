#!/usr/bin/env bash
# Launcher: scp the M3 q4_0 de-reference-form patch + board runner onto k1, start under nohup
# (survives ssh disconnect; poll /tmp/g6m3q40/run.log for ALL_DONE_q40_m3).
set -uo pipefail
BOARD="${BOARD:-k1}"
BD="/tmp/g6m3q40"
H="tools/e2e-harness/board/g6-m3-ime-deref"
ssh "$BOARD" "mkdir -p $BD"
scp "$H/forward-route-patch-q40-deref.py" "$BOARD:$BD/" >/dev/null
scp "$H/run-m3-q40-board.sh" "$BOARD:$BD/" >/dev/null
ssh "$BOARD" "cd $BD && rm -f run.log && PP='${PP:-32}' REPS='${REPS:-4}' PASSES='${PASSES:-3}' THREADS='${THREADS:-4}' nohup bash run-m3-q40-board.sh > run.log 2>&1 & echo launched pid=\$!"
echo "launched q40 M3; poll $BD/run.log for ALL_DONE_q40_m3"
