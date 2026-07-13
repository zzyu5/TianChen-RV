#!/usr/bin/env bash
# [G6-B Phase-2 q2_K @ k1] LOAD-GATE then measure. Runs ON THE BOARD. Waits until no foreign
# llama-bench/llama-cli/perf measurement is active AND 1-min loadavg is at quiet baseline, THEN
# runs the 3-way phase-split. Avoids co-measuring with G6-A M2 (whose load would poison timing).
set -u
WK=/tmp/g6b-p2-q2k
MAXWAIT="${MAXWAIT:-2400}"   # up to 40 min waiting for a quiet window
LOADCAP="${LOADCAP:-2.0}"    # 1-min loadavg ceiling: measure only when board is quiet (coordinator: no measure at loadavg>2)
POLL="${POLL:-30}"
SELFPID=$$
waited=0
while :; do
  # foreign measurement processes (exclude this script's own tree)
  FOREIGN=$(pgrep -f 'llama-bench|llama-cli|perf stat|perf record' 2>/dev/null | grep -vw "$SELFPID" | wc -l)
  L1=$(awk '{print $1}' /proc/loadavg)
  quiet=$(awk -v l="$L1" -v c="$LOADCAP" 'BEGIN{print (l<c)?1:0}')
  if [ "$FOREIGN" -eq 0 ] && [ "$quiet" -eq 1 ]; then
    echo "[gate] CLEAR after ${waited}s: foreign_bench=$FOREIGN loadavg1=$L1 (<$LOADCAP) -> measuring"
    break
  fi
  echo "[gate] WAIT ${waited}s: foreign_bench=$FOREIGN loadavg1=$L1 (cap $LOADCAP) -- board busy, re-poll in ${POLL}s"
  if [ "$waited" -ge "$MAXWAIT" ]; then echo "[gate] MAXWAIT ${MAXWAIT}s reached -- proceeding anyway (record load in fingerprint)"; break; fi
  sleep "$POLL"; waited=$((waited+POLL))
done
echo "[gate] proceeding with phase-split at loadavg=$(cat /proc/loadavg)"
exec bash "$WK/g6b_p2_phase_split.sh"
