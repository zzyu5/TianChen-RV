#!/usr/bin/env bash
# run_ondevice_verify.sh -- reusable on-device (ssh rvv / ssh k1) verification
# harness for M-FLAT typed flat block-dot kernels.
#
# Pipeline (mirrors the gate4-*-ssh evidence flow):
#   1. export the REAL RISC-V .o locally via tcrv-opt (front door + emission
#      plans) | tcrv-translate --tcrv-export-target-artifact
#   2. scp the .o + the C driver to the target board
#   3. capture the board profile (uname / clang version / lscpu)
#   4. compile the driver + .o on-board, TWICE:
#        - march=rv64gcv  (RVV; the reference loop is clang-autovectorized)
#        - march=rv64gc   (no V; the reference loop is pure scalar)
#   5. run each; the driver does bit-exact numerical verify + perf
#   6. copy all stdout back into experiments/<OUT_SUBDIR>/
#
# REUSE for the next flat格 (q4_0/q4_1/q5_0/q5_1): set the vars below.
#   TEST_MLIR      : the *-full-pipeline-export-e2e.mlir for that格
#   FRONT_DOOR_PASS: --tcrv-rvv-materialize-<格>-block-dot-source-front-door
#   DRIVER         : the格's *_verify_driver.c (copy q8_0's, change decode+quant)
#   OUT_SUBDIR     : experiments/<name>
#   TARGETS        : ssh host aliases to run on (e.g. "rvv" or "rvv k1")
set -u

# ---------------- configurable (q8_0 defaults) ----------------
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TEST_MLIR="${TEST_MLIR:-$REPO/test/Target/RVV/q8-0-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir}"
FRONT_DOOR_PASS="${FRONT_DOOR_PASS:---tcrv-rvv-materialize-q8-0-q8-0-block-dot-source-front-door}"
DRIVER="${DRIVER:-$REPO/.trellis/scripts/ondevice/q8_0_verify_driver.c}"
OUT_SUBDIR="${OUT_SUBDIR:-ondevice-q8_0}"
TARGETS="${TARGETS:-rvv}"
KERNEL_LABEL="${KERNEL_LABEL:-q8_0_q8_0_flat_block_dot}"
# driver args: n trials perf_iters
DRV_N="${DRV_N:-4096}"
DRV_TRIALS="${DRV_TRIALS:-256}"
DRV_ITERS="${DRV_ITERS:-200000}"

TCRV_OPT="$REPO/build/bin/tcrv-opt"
TCRV_TR="$REPO/build/bin/tcrv-translate"
OUTDIR="$REPO/experiments/$OUT_SUBDIR"
mkdir -p "$OUTDIR"

echo "== step 1: export real RISC-V .o locally =="
OBJ="$OUTDIR/kernel_${KERNEL_LABEL}.o"
"$TCRV_OPT" "$TEST_MLIR" "$FRONT_DOOR_PASS" --tcrv-materialize-emission-plans 2>"$OUTDIR/export.err" \
  | "$TCRV_TR" --tcrv-export-target-artifact > "$OBJ" 2>>"$OUTDIR/export.err"
if [ ! -s "$OBJ" ]; then
  echo "FATAL: .o export failed; see $OUTDIR/export.err"; sed -n '1,20p' "$OUTDIR/export.err"; exit 1
fi
echo "exported $(basename "$OBJ") ($(stat -c%s "$OBJ") bytes)"
file "$OBJ"

DRIVER_BASE="$(basename "$DRIVER")"

for HOST in $TARGETS; do
  echo ""
  echo "======================================================================"
  echo "== target: ssh $HOST =="
  echo "======================================================================"
  RDIR="/tmp/tcrv_ondevice_${OUT_SUBDIR}_$$"
  HOUT="$OUTDIR/host_$HOST"
  mkdir -p "$HOUT"

  ssh "$HOST" "mkdir -p $RDIR" || { echo "SKIP $HOST: ssh failed"; continue; }
  scp -q "$OBJ" "$DRIVER" "$HOST:$RDIR/" || { echo "SKIP $HOST: scp failed"; continue; }

  # ---- board profile ----
  ssh "$HOST" "bash -s" > "$HOUT/target_profile.txt" 2>&1 <<PROFILE
echo "ssh_target=$HOST"
echo "remote_uname=\$(uname -a)"
CLANG=\$(command -v clang-17 || command -v clang)
echo "clang_path=\$CLANG"
echo "clang_version=\$(\$CLANG --version | head -1)"
echo "vlenb_hint=\$(cat /proc/cpuinfo 2>/dev/null | grep -i -m1 vlen || echo 'n/a')"
echo "lscpu_begin"; lscpu 2>/dev/null | grep -iE 'arch|cpu\(s\)|model|flags|vlen' ; echo "lscpu_end"
PROFILE
  echo "-- board profile ($HOST) --"; cat "$HOUT/target_profile.txt"

  # ---- compile + run for each march ----
  for MARCH in rv64gcv rv64gc; do
    echo "-- compile+run march=$MARCH on $HOST --"
    ssh "$HOST" "bash -s" > "$HOUT/run_${MARCH}.txt" 2>&1 <<REMOTE
set -e
cd $RDIR
CLANG=\$(command -v clang-17 || command -v clang)
# --rtlib=compiler-rt: the kernel .o and the reference both need the soft
# fp16<->fp32 builtins (__extendhfsf2/__truncsfhf2); the compiler-rt builtins
# archive has them, the board's default gcc-12 libgcc does not.
FLAGS="-O2 -march=$MARCH -mabi=lp64d -ffp-contract=off --rtlib=compiler-rt"
echo "compile_flags=\$FLAGS  clang=\$CLANG"
\$CLANG \$FLAGS "$DRIVER_BASE" "$(basename "$OBJ")" -lm -o driver_${MARCH} 2>&1 || {
  echo "COMPILE_FAILED march=$MARCH"; exit 20; }
./driver_${MARCH} $MARCH $DRV_N $DRV_TRIALS $DRV_ITERS
echo "exit_status=\$?"
REMOTE
    cat "$HOUT/run_${MARCH}.txt"
  done

  ssh "$HOST" "rm -rf $RDIR" 2>/dev/null || true
done

echo ""
echo "== results saved under $OUTDIR/ =="
ls -R "$OUTDIR"
