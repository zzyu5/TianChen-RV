#!/usr/bin/env bash
# run_q4k_handbrick_msweep_k1.sh — G7 §3 M-sweep: q4_K@k1 REPACK-GEMM (ours vl=8 core s6_q4K.c,
#   md5 90d454da) vs TRUE shipped hand-brick ggml_gemm_q4_K_16x1_q8_K. BOTH are repack-GEMM that
#   reuse weight across the col/row group => both compute-bound; M-sweep tests if ours closes the
#   0.622x gap at higher M. Pure RE-RUN of the already-built board binary (/tmp/q4k_hb_resolve/kqhb_O2)
#   at nr{4,8,16,32}. NOTE: this is the vl=8 core (kernel-sym LOSS), NOT the sealed vl=16 winner
#   (e437fd3b source unavailable). Symmetric clang-18, VLEN256. NO git, main tree untouched.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NRGRID="${NRGRID:-4 8 16 32}"
K="${K:-2048}"; NC="${NC:-512}"; POOL="${POOL:-8}"; ROUNDS="${ROUNDS:-12}"; ITERS="${ITERS:-20}"; CORE="${CORE:-3}"

REMOTE=/tmp/msweep_q4k_hb_remote.sh
cat > "$REMOTE" <<REMOTE_EOF
#!/usr/bin/env bash
set -uo pipefail
RDIR=/tmp/q4k_hb_resolve; K=$K; NC=$NC; POOL=$POOL; ROUNDS=$ROUNDS; ITERS=$ITERS; CORE=$CORE
NRGRID="$NRGRID"
REMOTE_EOF
cat >> "$REMOTE" <<'REMOTE_EOF'
cd "$RDIR" || { echo "SCRATCH_MISSING"; exit 40; }
GGML=/data/k1build-stock/bin
LOG=$RDIR/msweep_run.log; : > "$LOG"
echo "# q4_K@k1 handbrick M-sweep  board=$(uname -srm)" | tee -a "$LOG"
echo "# opp_lib=$(readlink -f $GGML/libggml-cpu.so) md5=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)" | tee -a "$LOG"
[ -x ./kqhb_O2 ] || { echo "BINARY_MISSING kqhb_O2 (need rebuild via run_k1_q4k_handbrick.sh)"; exit 41; }
echo "# ours_kernel_md5=$(md5sum s6_q4K.c 2>/dev/null|cut -d' ' -f1) (vl=8 core; NOT sealed vl=16)" | tee -a "$LOG"
echo "# governor(core$CORE)=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null) freq=$(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null)" | tee -a "$LOG"
# load-gate: wait for loadavg<2.5
for i in $(seq 1 40); do la=$(cut -d' ' -f1 /proc/loadavg); ok=$(awk -v l="$la" 'BEGIN{print (l<2.5)?1:0}'); [ "$ok" = 1 ] && break; sleep 3; done
echo "# loadavg_begin=$(cat /proc/loadavg)" | tee -a "$LOG"
for NR in $NRGRID; do
  echo "===== nr=$NR (M) =====" | tee -a "$LOG"
  LD_LIBRARY_PATH="$GGML" taskset -c $CORE ./kqhb_O2 $K $NR $NC $POOL $ROUNDS $ITERS 0xC0FFEE 2>&1 | tee -a "$LOG"
done
echo "# loadavg_end=$(cat /proc/loadavg)" | tee -a "$LOG"
pkill -u $(id -un) -f kqhb_O2 2>/dev/null
echo "# STRAY=$(pgrep -c -f kqhb_O2 || echo 0)" | tee -a "$LOG"
echo "# ALL_DONE" | tee -a "$LOG"
REMOTE_EOF

scp -q "$REMOTE" k1:/tmp/q4k_hb_resolve/msweep_remote.sh 2>/dev/null || { echo "scp failed (scratch missing?)"; }
ssh k1 "bash /tmp/q4k_hb_resolve/msweep_remote.sh"
mkdir -p "$HERE/raw"
scp -q k1:/tmp/q4k_hb_resolve/msweep_run.log "$HERE/raw/k1_q4k_handbrick_msweep.log" 2>/dev/null && echo "[q4k-hb-msweep] log pulled"
