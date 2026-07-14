#!/usr/bin/env bash
# run_vecdot_msweep_k1.sh — G7 §3 M-sweep batch-regime: vec_dot(block-dot) @k1 M-sweep.
# Both sides block-dot (ours weft-emitted clang-18 vs opp as-shipped ggml_vec_dot). Both re-stream
# weight per row => intensity flat in M (control lane). M grid {1,2,4,8,16,32}. Symmetric clang-18,
# VLEN256, cold 224MiB flush, load-gated core 0-3. Reuses SEALED driver+kernels from ../../vecdot-k1.
# NOTE: k1 login shell is zsh -> we ship the remote body as a FILE and run it explicitly under bash.
# NO git, main tree untouched, stock lib read-only.
set -uo pipefail
SRC="/home/kingdom/phdworks/TianchenRV/experiments/active/g7-census/vecdot-k1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RDIR=/tmp/g7_msweep_vecdot_k1
K="${K:-2048}"; NC="${NC:-512}"; HITERS="${HITERS:-8}"; REPS="${REPS:-12}"
MGRID="${MGRID:-1 2 4 8 16 32}"
FMTS="q2_K q3_K q4_K q5_K q6_K q4_0 q4_1 q5_0 q5_1 q8_0"

# ---- build the remote bash body (quoted heredoc: NO local expansion; config injected via header) ----
REMOTE=/tmp/msweep_vecdot_k1_remote.sh
cat > "$REMOTE" <<REMOTE_EOF
#!/usr/bin/env bash
set -uo pipefail
RDIR=$RDIR; K=$K; NC=$NC; HITERS=$HITERS; REPS=$REPS
MGRID="$MGRID"; FMTS="$FMTS"
REMOTE_EOF
cat >> "$REMOTE" <<'REMOTE_EOF'
cd "$RDIR"
MARCH=rv64gcv_zfh_zvfh_zicbop_zihintpause
GGML=/data/k1build-stock/bin
CC=clang-18
LOG=$RDIR/run.log; : > "$LOG"
echo "# BUILD vecdot-msweep@k1 CC=$($CC --version|head -1) march=$MARCH" | tee -a "$LOG"
echo "# board=$(uname -srm) ggml_so_md5=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)" | tee -a "$LOG"
ls $GGML/libggml-cpu.so >/dev/null || { echo GGML_MISSING; exit 40; }
for f in $FMTS; do
  $CC -O3 -march=$MARCH -mabi=lp64d -x c++ kernels/${f}.kernel.c -c -o k_${f}.o 2>cc_${f}.err || { echo "KERN_FAIL $f"; sed -n '1,6p' cc_${f}.err; exit 3; }
done
$CC -O2 -march=$MARCH -mabi=lp64d -x c vecdot_census_driver.c -c -o drv.o 2>cc_drv.err || { echo DRV_FAIL; sed -n '1,12p' cc_drv.err; exit 4; }
$CC drv.o k_*.o -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o vecdot_census 2>ld.err || { echo LINK_FAIL; sed -n '1,12p' ld.err; exit 5; }
echo "# linked OK -> ./vecdot_census" | tee -a "$LOG"

# ---- load-gate: pick idle core in 0-3 ----
echo "# loadavg_begin=$(cat /proc/loadavg)" | tee -a "$LOG"
read_busy(){ awk -v c="cpu$1" '$1==c{idle=$5+$6; tot=$2+$3+$4+$5+$6+$7+$8; print tot" "idle}' /proc/stat; }
declare -A B0 I0
for c in 0 1 2 3; do read t i < <(read_busy $c); B0[$c]=$t; I0[$c]=$i; done
sleep 0.4
BESTC=-1; BESTIDLE=-1
for c in 0 1 2 3; do read t i < <(read_busy $c); dt=$((t-${B0[$c]})); di=$((i-${I0[$c]}));
  pct=$(( dt>0 ? 100*di/dt : 0 )); if [ $pct -gt $BESTIDLE ]; then BESTIDLE=$pct; BESTC=$c; fi; done
if [ $BESTIDLE -lt 70 ]; then echo "# LOAD_GATE_FAIL best core$BESTC idle=${BESTIDLE}% (<70) ABORT" | tee -a "$LOG"; exit 9; fi
CORE=$BESTC
echo "# LOAD_GATE_OK pin core=$CORE idle=${BESTIDLE}% gov=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null)" | tee -a "$LOG"
run(){ LD_LIBRARY_PATH=$GGML taskset -c $CORE ./vecdot_census "$@" 2>>"$LOG" | tee -a "$LOG"; }

echo "=== [V] VERIFY byte-exact (M=1 & M=32, nc=64) ===" | tee -a "$LOG"
VFAIL=0
for f in $FMTS; do run $f $K 1 64 1 10 0xBEEF 1 || VFAIL=1; run $f $K 32 64 1 10 0xBEEF 1 || VFAIL=1; done
echo "# VERIFY_FAIL=$VFAIL" | tee -a "$LOG"

echo "=== [S] M-SWEEP GEVM/block-dot ===" | tee -a "$LOG"
for M in $MGRID; do
  for f in $FMTS; do SEED=$(printf '0x%X' $((0x1000 + M*97 + RANDOM))); run $f $K $M $NC $HITERS $REPS $SEED; done
  echo "# loadavg_after_M${M}=$(cat /proc/loadavg)" | tee -a "$LOG"
done
pkill -u $(id -un) -f vecdot_census 2>/dev/null
echo "# STRAY=$(pgrep -c -f vecdot_census || echo 0)" | tee -a "$LOG"
echo "# loadavg_end=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "# ALL_DONE" | tee -a "$LOG"
REMOTE_EOF

echo "[msweep-vecdot-k1] scp harness + remote script -> k1:$RDIR"
ssh k1 "mkdir -p $RDIR/kernels"
scp -q "$SRC/vecdot_census_driver.c" k1:$RDIR/
for f in $FMTS; do scp -q "$SRC/kernels/${f}.kernel.c" k1:$RDIR/kernels/; done
scp -q "$REMOTE" k1:$RDIR/remote.sh
ssh k1 "bash $RDIR/remote.sh"
mkdir -p "$HERE/raw"
scp -q k1:$RDIR/run.log "$HERE/raw/k1_vecdot_msweep.log" 2>/dev/null && echo "[msweep-vecdot-k1] log pulled"
