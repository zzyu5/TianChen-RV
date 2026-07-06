#!/usr/bin/env bash
# run_seal_measure.sh -- board-side L1-SEAL measurement orchestrator. Waits for
# the full-capability rebuild (ALL.done), then runs, in one sitting:
#   (1) strict preflight (gate-1 NON-advisory: ALLOW_MARCH_INCOMPLETE=0)
#   (2) paired A/B phase-split (prefill pp128 + decode tg32)
#   (3) decode snapshots at 3 board-memory-pressure levels (light/medium/heavy)
#   (4) objdump mechanism seal + symmetry libcall scan
#   (5) greedy-token correctness gate
# All outputs land in $RDIR/results/. Idempotent per-step (overwrites outputs).
set -u
RDIR="${RDIR:-/tmp/tcrv_l1seal}"
OUT="$RDIR/results"; mkdir -p "$OUT"
A_BUILD=/home/ubuntu/tcrv-llamacpp/build-gcc15-rv64gcv
B_BUILD=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv
MODEL=/home/ubuntu/tcrv-llamacpp/models/tinyllama-q4_0.gguf
export LDPATH=/opt/tcrv-toolchains/gcc-15.2.0/lib
CORES=8-11; THREADS=4
common="A_BUILD=$A_BUILD B_BUILD=$B_BUILD MODEL=$MODEL LDPATH=$LDPATH"

echo "== wait for rebuild ALL.done =="
until [ -f "$RDIR/ALL.done" ]; do sleep 8; done
grep -q "FATAL" "$RDIR/rebuild2.log" && { echo "REBUILD HAD FATAL -- abort"; exit 9; }
echo "rebuild complete; both bins:"
ls -l --time-style=+%H:%M "$A_BUILD/bin/llama-bench" "$B_BUILD/bin/llama-bench"

echo ""; echo "############ (1) STRICT PREFLIGHT (gate-1 non-advisory) ############"
env $common EXP_VLEN=128 PROBE_MODEL=$MODEL \
  ALLOW_MARCH_INCOMPLETE_IF_LIBCALL_CLEAN=0 \
  bash "$RDIR/preflight_e2e.sh" 2>&1 | tee "$OUT/preflight.txt"
if ! grep -q "PREFLIGHT PASS" "$OUT/preflight.txt"; then
  echo "STRICT PREFLIGHT FAILED -- stopping (fail-closed)"; touch "$RDIR/MEASURE.done"; exit 30
fi

echo ""; echo "############ (2) PHASE-SPLIT A/B (prefill pp128 + decode tg32) ############"
env $common CORES=$CORES THREADS=$THREADS PP=128 TG=32 REPS=4 PASSES=2 \
  bash "$RDIR/phase_split_ab.sh" 2>&1 | tee "$OUT/phase_split_raw.txt"

echo ""; echo "############ (3) DECODE SNAPSHOTS (light/medium/heavy) ############"
env $common MEAS_CORES=$CORES THREADS=$THREADS TG=32 REPS=4 PASSES=2 \
  LEVEL=light  LOAD_WORKERS=0  LOAD_CORES=12-63 \
  bash "$RDIR/decode_snapshots.sh" 2>&1 | tee "$OUT/decode_light.txt"
env $common MEAS_CORES=$CORES THREADS=$THREADS TG=32 REPS=4 PASSES=2 \
  LEVEL=medium LOAD_WORKERS=12 LOAD_CORES=12-63 \
  bash "$RDIR/decode_snapshots.sh" 2>&1 | tee "$OUT/decode_medium.txt"
env $common MEAS_CORES=$CORES THREADS=$THREADS TG=32 REPS=4 PASSES=2 \
  LEVEL=heavy  LOAD_WORKERS=40 LOAD_CORES=12-63 \
  bash "$RDIR/decode_snapshots.sh" 2>&1 | tee "$OUT/decode_heavy.txt"

echo ""; echo "############ (4) OBJDUMP MECHANISM SEAL ############"
env A_BUILD=$A_BUILD B_BUILD=$B_BUILD OUTDIR=$OUT \
  bash "$RDIR/objdump_seal.sh" 2>&1 | tee "$OUT/objdump_seal.txt"

echo ""; echo "############ (5) CORRECTNESS GATE (greedy-token) ############"
env $common CORES=8-23 NTOK=24 THREADS=16 \
  bash "$RDIR/correctness_gate.sh" 2>&1 | tee "$OUT/correctness.txt" || true

echo ""; echo "== ALL MEASUREMENT DONE =="; touch "$RDIR/MEASURE.done"
