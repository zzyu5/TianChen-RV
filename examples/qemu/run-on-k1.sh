#!/usr/bin/env bash
# Real-hardware proof for an RVV slice on a SpacemiT X60 board (ssh alias `k1`).
# This is the canonical tier-3 proof: compile the compiler-GENERATED kernel + the
# student harness natively on real RVV 1.0 (+Zvfh) silicon, run, compare vs the
# scalar oracle baked into the harness. A clean run prints "... proof ok".
#
# Usage:
#   examples/qemu/run-on-k1.sh <generated.cpp> <harness.cpp> [case-name] [march]
# Example (the worked example, already verified):
#   examples/qemu/run-on-k1.sh add_generated.reference.cpp harness_add.cpp add
#
# Requires: an ssh host alias `k1` reachable non-interactively (BatchMode), with
# clang++ on it. (The maintainer's classroom board: clang++ 18, -march=rv64gcv.)
set -euo pipefail

GEN=${1:?need generated .cpp}
HARNESS=${2:?need harness .cpp}
CASE=${3:-rvv_case}
MARCH=${4:-rv64gcv}
HOST=${K1_HOST:-k1}
REMOTE_DIR=${K1_DIR:-/tmp/tcrv-classroom-$CASE}

echo ">> [$HOST] mkdir $REMOTE_DIR"
ssh -o BatchMode=yes -o ConnectTimeout=8 "$HOST" "mkdir -p '$REMOTE_DIR'"
echo ">> scp $GEN $HARNESS -> $HOST:$REMOTE_DIR"
scp -o BatchMode=yes -o ConnectTimeout=8 "$GEN" "$HARNESS" "$HOST:$REMOTE_DIR/"
echo ">> [$HOST] clang++ -O2 -march=$MARCH  (native build on real RVV hardware)"
ssh -o BatchMode=yes -o ConnectTimeout=8 "$HOST" \
  "cd '$REMOTE_DIR' && clang++ -std=c++17 -O2 -march=$MARCH -o '$CASE' '$(basename "$GEN")' '$(basename "$HARNESS")' && ./'$CASE'"
echo ">> done (a 'proof ok' line above == real-hardware byte-exact pass)"
