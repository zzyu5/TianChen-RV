#!/usr/bin/env bash
# run_e2e.sh -- LOCAL orchestrator for the T6 e2e phase-split harness.
# Ships the board/ scripts to the target, runs the fail-closed preflight, then
# (only if green) the paired A/B phase-split + the correctness gate, pulls the
# raw logs back into results/<LABEL>/, and aggregates them locally.
#
# The A/B is two llama.cpp build trees already on the board:
#   A (ours)  = patched-ggml tree calling the TianChen-RV compiler-emitted repack
#               kernel (engages at VLEN128).
#   B (stock) = unpatched upstream llama.cpp (ggml block-dot at VLEN128).
#
# Pinned rvv (openEuler VLEN128) defaults below; override via env for other
# boards / models / quant formats (q8_0, etc.) -- see models.manifest.csv.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOST="${HOST:-rvv}"
EXP_VLEN="${EXP_VLEN:-128}"
BOARD_LABEL="${BOARD_LABEL:-rvv-openeuler-vlen128}"
CLANG_LABEL="${CLANG_LABEL:-gcc-15.2.0-rv64gcv}"   # board+compiler stamp for results
LDPATH="${LDPATH:-/opt/tcrv-toolchains/gcc-15.2.0/lib}"
A_BUILD="${A_BUILD:-/home/ubuntu/tcrv-llamacpp/build-gcc15-rv64gcv}"
B_BUILD="${B_BUILD:-/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv}"
MODEL="${MODEL:-/home/ubuntu/tcrv-llamacpp/models/tinyllama-q4_0.gguf}"
QUANT="${QUANT:-q4_0}"
CORES="${CORES:-8-11}"; THREADS="${THREADS:-4}"
PP="${PP:-128}"; TG="${TG:-32}"; REPS="${REPS:-5}"; PASSES="${PASSES:-2}"
NTOK="${NTOK:-24}"
ALLOW_MARCH_INCOMPLETE_IF_LIBCALL_CLEAN="${ALLOW_MARCH_INCOMPLETE_IF_LIBCALL_CLEAN:-1}"
LABEL="${LABEL:-${BOARD_LABEL}-${QUANT}-$(date +%Y%m%d-%H%M)}"
OUT="$HERE/results/$LABEL"
mkdir -p "$OUT"

RDIR="/tmp/tcrv_e2e_$$"
common_env="A_BUILD='$A_BUILD' B_BUILD='$B_BUILD' MODEL='$MODEL' LDPATH='$LDPATH' CORES='$CORES' THREADS='$THREADS'"

echo "== ship board scripts to $HOST:$RDIR =="
ssh "$HOST" "mkdir -p $RDIR" || { echo "FATAL: ssh $HOST failed"; exit 1; }
scp -q "$HERE/board/"*.sh "$HOST:$RDIR/" || { echo "FATAL: scp failed"; exit 1; }

echo ""
echo "== (1) PREFLIGHT =="
ssh "$HOST" "cd $RDIR && $common_env EXP_VLEN='$EXP_VLEN' PROBE_MODEL='$MODEL' \
  ALLOW_MARCH_INCOMPLETE_IF_LIBCALL_CLEAN='$ALLOW_MARCH_INCOMPLETE_IF_LIBCALL_CLEAN' \
  bash preflight_e2e.sh" 2>&1 | tee "$OUT/preflight.txt"
if ! grep -q "PREFLIGHT PASS" "$OUT/preflight.txt"; then
  echo "PREFLIGHT FAILED -> no measurement produced (fail-closed). See $OUT/preflight.txt"
  ssh "$HOST" "rm -rf $RDIR" 2>/dev/null; exit 30
fi

echo ""
echo "== (2) PHASE-SPLIT A/B (prefill pp$PP + decode tg$TG; $PASSES passes x $REPS reps) =="
ssh "$HOST" "cd $RDIR && $common_env PP='$PP' TG='$TG' REPS='$REPS' PASSES='$PASSES' \
  bash phase_split_ab.sh" 2>&1 | tee "$OUT/phase_split_raw.txt"

echo ""
echo "== (3) CORRECTNESS GATE (greedy-token consistency + logits sanity) =="
ssh "$HOST" "cd $RDIR && $common_env NTOK='$NTOK' CORES='${CG_CORES:-8-23}' \
  bash correctness_gate.sh" 2>&1 | tee "$OUT/correctness.txt" || true

ssh "$HOST" "rm -rf $RDIR" 2>/dev/null || true

echo ""
echo "== (4) AGGREGATE (local) =="
python3 "$HERE/aggregate_e2e.py" "$OUT/phase_split_raw.txt" \
  --board "$BOARD_LABEL/$CLANG_LABEL" --model "$(basename "$MODEL")" --quant "$QUANT" \
  --out "$OUT/evidence.json" 2>&1 | tee "$OUT/aggregate.txt"

echo ""
echo "== results in $OUT/ =="
ls -1 "$OUT"
