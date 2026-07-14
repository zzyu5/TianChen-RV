#!/usr/bin/env bash
# G8 §六.3 [BUG-K1-Q2K-GEMM-WRONG] ZERO-MODEL adjudication on k1.
# ours' ACTUAL compiled q2_K GEMM kernel vs one independent canonical ref, on BOTH
# ours-straight layout and ggml-stock permuted layout. stock .so read-only. NO git.
set -uo pipefail
RDIR=/tmp/g8k1_q2kadj
CC=/usr/bin/clang
MARCH=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs_zvl256b
GGML=/data/k1build-stock/bin
KSRC=/tmp/g8k1_gemm_census/kernels
SEAL=$RDIR/build_seal.txt; LOG=$RDIR/run.log
mkdir -p $RDIR; : > "$SEAL"; : > "$LOG"
echo "# q2k-adjudicate CC=$($CC --version|head -1) march=$MARCH -O2 $(date -u +%FT%TZ)" | tee -a "$SEAL"
echo "# stock_so_md5=$(md5sum $GGML/libggml-cpu.so|cut -d' ' -f1)" | tee -a "$SEAL"

src=$KSRC/repack_gemm_q2_K_q8_K_vlen256_unrolled.kernel.c
[ -f "$src" ] || { echo "MISSING census kernel $src"; exit 3; }
$CC -O2 -march=$MARCH -mabi=lp64d -ffp-contract=on -x c++ "$src" -c -o $RDIR/k_q2k.o 2>$RDIR/cc_k.err \
  || { echo KERN_FAIL; head -8 $RDIR/cc_k.err; exit 3; }
$CC -O2 -march=$MARCH -mabi=lp64d $RDIR/q2k_adjudicate.cpp -c -o $RDIR/adj.o 2>$RDIR/cc_h.err \
  || { echo HARNESS_FAIL; head -30 $RDIR/cc_h.err; exit 4; }
$CC $RDIR/adj.o $RDIR/k_q2k.o -L$GGML -Wl,-rpath,$GGML -lggml-cpu -lggml-base -lggml -lstdc++ -lm -o $RDIR/adj 2>$RDIR/ld.err \
  || { echo LINK_FAIL; head -30 $RDIR/ld.err; exit 5; }
echo "# linked OK -> adj" | tee -a "$SEAL"

echo "# loadavg=$(cat /proc/loadavg)" | tee -a "$LOG"
read_busy(){ awk -v c="cpu$1" '$1==c{idle=$5+$6;tot=$2+$3+$4+$5+$6+$7+$8;print tot" "idle}' /proc/stat; }
declare -A B0 I0; for c in 0 1 2 3; do read t i < <(read_busy $c); B0[$c]=$t; I0[$c]=$i; done
sleep 0.4; BESTC=0; BESTIDLE=-1
for c in 0 1 2 3; do read t i < <(read_busy $c); dt=$((t-${B0[$c]})); di=$((i-${I0[$c]}));
  pct=$(( dt>0?100*di/dt:0 )); [ $pct -gt $BESTIDLE ] && { BESTIDLE=$pct; BESTC=$c; }; done
echo "# PIN core=$BESTC idle=${BESTIDLE}%" | tee -a "$LOG"
run(){ LD_LIBRARY_PATH=$GGML taskset -c "$BESTC" "$RDIR/adj" "$@" 2>>"$LOG" | tee -a "$LOG"; }

echo "======== ADJUDICATE ========" | tee -a "$LOG"
run 2048 8 256 0x2718 | tee -a "$LOG"
echo "----" | tee -a "$LOG"
run 2048 4 32 0x1337 | tee -a "$LOG"
echo "----" | tee -a "$LOG"
run 512 16 256 0xBEEF | tee -a "$LOG"
echo "# loadavg_after=$(cat /proc/loadavg)" | tee -a "$LOG"
echo "# DONE" | tee -a "$LOG"
