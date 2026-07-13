#!/usr/bin/env bash
# G6-A M7 (vmadot-tiling / array-utilization) -- q4_0 IME bridge, BOARD-SIDE runner (k1).
# Single patch/build/restore cycle on top of the M6 (B-feed locality) bridge. Runs:
#   [X-0/array-util] WIDE vmadot output tiling decomposition (4-hart, ns_matmul = parallel crit path,
#                    ith==0 timed). Profiling configs (NJOUTER base, PARSETUP):
#                      compute_l1     = width-1 vmadot from FIXED L1 scratch   -> compute only (M6 residual)
#                      compute_l1_w2  = WIDE-2 vmadot from FIXED L1 scratch    -> wide compute only
#                      compute_l1_w4  = WIDE-4 vmadot from FIXED L1 scratch    -> wide compute only
#                      onjout_noepi   = width-1 nj-outer vmadot + real feed
#                      onw2_noepi     = WIDE-2  nj-outer vmadot + real feed
#                      onjout_full    = M6 nj-outer full vec (width-1)         -> t_M6 matmul
#                      onw2_full      = M7 nj-outer full vec WIDE-2            -> t_M7w2 matmul
#                      onw4_full      = M7 nj-outer full vec WIDE-4            -> t_M7w4 matmul
#                    array-util win = compute_l1 - compute_l1_wN.
#   [correctness] greedy A==B (HARD GATE): onw2 AND onw4 vs onjout MUST be byte-identical (wide tiling
#                 reuses A in-register across NJW col-tiles; each 4x4 int32 sub-tile accumulates its
#                 fragments in the SAME kf order into its OWN accumulator -> bit-exact). vs off =
#                 in-family coherence; vs ven = known near-tie flip. All md5 == f5e77482 (M1..M6 seal).
#   [perf] interleaved paired llama-bench, pp32 prefill, same -t:
#          off/ven/onjout/onw2/onw4 : every pass (>=12 samples/side)
#          off    = build-off (stock RVV)          ; ven = build-ime env unset (vendor IME)
#          onjout = M6 (BRIDGE CACHE THREADS DEREF DEREF_EPI PARSETUP EPIVEC NJOUTER)
#          onw2   = M7 (+ TILEW=2)  ; onw4 = M7 (+ TILEW=4)
# Our-N-hart vs vendor-N-hart. decode (M=1) never routed. ALWAYS restores board (md5 double-proof).
set -uo pipefail
DIR="/home/bianbu/tcrv-k1-llama"
IME="$DIR/ggml/src/ggml-cpu/spacemit/ime.cpp"
MODEL="${MODEL:-$DIR/models/tinyllama-q4_0.gguf}"
BD="/tmp/g6m7q40"
BASE_IME="40962c7e7c732bf472ae88cef89ced8d"
BASE_SO="71cc4d295dac29382a0a7d4d5bd0c425"
SEAL_MD5="f5e77482dbd78bb0543b9a64c1c2f29c"   # M1..M6 greedy-output seal
BENCH_IME="$DIR/build-ime/bin/llama-bench"
BENCH_OFF="$DIR/build-off/bin/llama-bench"
CMPL_IME="$DIR/build-ime/bin/llama-completion"
CMPL_OFF="$DIR/build-off/bin/llama-completion"
PP="${PP:-32}"; REPS="${REPS:-4}"; PASSES="${PASSES:-3}"; THREADS="${THREADS:-4}"
CORES="${CORES:-0-3}"; CORE0=0
ENVV="TCRV_IME_Q40_BRIDGE"; CACHEV="TCRV_IME_Q40_CACHE"; THREADSV="TCRV_IME_Q40_THREADS"
DEREFV="TCRV_IME_Q40_DEREF"; DEREFEPIV="TCRV_IME_Q40_DEREF_EPI"; PARV="TCRV_IME_Q40_PARSETUP"
EPIVECV="TCRV_IME_Q40_EPIVEC"; NJOUTV="TCRV_IME_Q40_NJOUTER"; TILEWV="TCRV_IME_Q40_TILEW"
NOEPIV="TCRV_IME_Q40_MMPROF_NOEPI"; L1V="TCRV_IME_Q40_MMPROF_L1"; PROFV="TCRV_IME_Q40_PROF"
MARK="TCRV-IME-Q40-BRIDGE"
PIN="taskset -c $CORES"
NJOUT_ENV="$ENVV=1 $CACHEV=1 $THREADSV=1 $DEREFV=1 $DEREFEPIV=1 $PARV=1 $EPIVECV=1 $NJOUTV=1"
W2_ENV="$NJOUT_ENV $TILEWV=2"
W4_ENV="$NJOUT_ENV $TILEWV=4"
mkdir -p "$BD"
cd "$DIR"

SO=$(readlink -f build-ime/bin/libggml-cpu.so)
cp -f "$IME" "$IME.ORIG"; cp -f "$SO" "$SO.ORIG"
IME_MD5=$(md5sum "$IME" | cut -d' ' -f1); SO_MD5=$(md5sum "$SO" | cut -d' ' -f1)
echo "baseline ime.cpp=$IME_MD5  so=$SO_MD5"
restore() {
  echo "=== RESTORE (trap) ==="
  cp -f "$IME.ORIG" "$IME" 2>/dev/null || true
  make -C build-ime ggml-cpu -j8 >>"$BD/route_build.log" 2>&1 && echo "restore-rebuild ok" || echo "restore-rebuild WARN"
  cp -f "$SO.ORIG" "$SO" 2>/dev/null || true
  NIME=$(md5sum "$IME" | cut -d' ' -f1); NSO=$(md5sum "$SO" | cut -d' ' -f1)
  { [ "$NIME" = "$BASE_IME" ] && [ "$NSO" = "$BASE_SO" ]; } && echo "RESTORE md5 ZERO-CHANGE OK (ime=$NIME so=$NSO)" || echo "RESTORE MD5 MISMATCH! ime=$NIME so=$NSO"
  echo "src_route_left=$(grep -c "$MARK" "$IME" 2>/dev/null || echo 0)"
  rm -f "$IME.ORIG" "$SO.ORIG"
  echo "litter_left=$(ls "$IME.ORIG" "$SO.ORIG" 2>/dev/null | wc -l)"
  echo "ALL_DONE_q40_m7"
}
trap restore EXIT INT TERM

echo "== ENV FINGERPRINT =="
echo "uname=$(uname -a)"
echo "isa=$(grep -m1 -i isa /proc/cpuinfo)"
echo "nproc=$(nproc) pinned_cores=$CORES threads=$THREADS"
echo "governor=$(cat /sys/devices/system/cpu/cpu$CORE0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
echo "cur_freq=$(cat /sys/devices/system/cpu/cpu$CORE0/cpufreq/scaling_cur_freq 2>/dev/null || echo NA)"
echo "loadavg=$(cat /proc/loadavg)"
echo "clang: $(clang --version 2>/dev/null | head -1)  (kernel==system: shipped clang-18 both builds)"
echo "model=$MODEL model_sha256=$(sha256sum "$MODEL" 2>/dev/null | cut -c1-16)"
echo "config: PP=$PP REPS=$REPS PASSES=$PASSES  sides=off/ven/onjout/onw2/onw4  prof=cl1/cl1w2/cl1w4/njnoepi/w2noepi/M6/w2/w4"

echo "=== patch + rebuild build-ime (q4_0 vmadot-tiling bridge) ==="
python3 "$BD/forward-route-patch-q40-tilew.py"
if make -C build-ime ggml-cpu -j8 >"$BD/route_build.log" 2>&1; then echo "rebuild ok"; else echo "BUILD FAIL"; tail -100 "$BD/route_build.log"; exit 1; fi
echo "vmadot_in_patched_so=$(objdump -d "$SO" | grep -c vmadot || true)  (baseline 32; M6=41; M7 adds w2(2)+w4(4)+wide njouter/noepi/l1 inlines)"

freq() { cat /sys/devices/system/cpu/cpu$CORE0/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

echo "== LOAD-GATE (static board check before measure) =="
echo "loadavg_pre=$(cat /proc/loadavg)"
echo "extern_jobs=$(pgrep -af 'llama-bench|llama-completion' | grep -v $$ | grep -vc pgrep || echo 0)"

echo "== WARMUP (dropped) =="
env $W2_ENV $PIN "$BENCH_IME" -m "$MODEL" -p 16 -n 0 -t "$THREADS" -r 1 >/dev/null 2>&1 || true

# ============================ [X-0 / array-util] WIDE-tiling compute decomposition ============
echo "======== PROFILE_BEGIN ========"
prof_run() { # $1=stem $2=envstr
  env $2 "$PROFV=1" $PIN "$BENCH_IME" -m "$MODEL" -p "$PP" -n 0 -t "$THREADS" -r 3 -o json \
      >"$BD/prof_$1.json" 2>"$BD/prof_$1.err" || true
  grep -E "TCRV-Q40-PROF|TCRV-IME-Q40-BRIDGE" "$BD/prof_$1.err" || echo "(no prof line $1!)"
}
echo "---- PROF compute_l1 (width-1 vmadot from FIXED L1: compute only) ----";      prof_run compute_l1    "$NJOUT_ENV $L1V=1"
echo "---- PROF compute_l1_w2 (WIDE-2 vmadot from FIXED L1: compute only) ----";     prof_run compute_l1_w2 "$W2_ENV $L1V=1"
echo "---- PROF compute_l1_w4 (WIDE-4 vmadot from FIXED L1: compute only) ----";     prof_run compute_l1_w4 "$W4_ENV $L1V=1"
echo "---- PROF onjout_noepi (width-1 nj-outer vmadot + real feed) ----";            prof_run onjout_noepi  "$NJOUT_ENV $NOEPIV=1"
echo "---- PROF onw2_noepi (WIDE-2 nj-outer vmadot + real feed) ----";               prof_run onw2_noepi    "$W2_ENV $NOEPIV=1"
echo "---- PROF onjout_full (M6 nj-outer full vec width-1) ----";                    prof_run onjout_full   "$NJOUT_ENV"
echo "---- PROF onw2_full (M7 nj-outer full vec WIDE-2) ----";                       prof_run onw2_full     "$W2_ENV"
echo "---- PROF onw4_full (M7 nj-outer full vec WIDE-4) ----";                       prof_run onw4_full     "$W4_ENV"
echo "======== PROFILE_END ========"

# ============================ [correctness] greedy A==B ============================
echo "======== CORRECTNESS_BEGIN ========"
CPROMPT="Once upon a time, in a small village nestled between two great mountains, there lived"
run_cmpl() { # $1=bin $2=env $3=stem
  env $2 $PIN "$1" -m "$MODEL" -p "$CPROMPT" -n 24 --temp 0 -s 0 -t "$THREADS" \
      -no-cnv --no-warmup --no-display-prompt --simple-io >"$3.out" 2>"$3.err" || true
}
run_cmpl "$CMPL_OFF" ""            "$BD/c_off"
run_cmpl "$CMPL_IME" ""            "$BD/c_ven"
run_cmpl "$CMPL_IME" "$NJOUT_ENV"  "$BD/c_onjout"
run_cmpl "$CMPL_IME" "$W2_ENV"     "$BD/c_onw2"
run_cmpl "$CMPL_IME" "$W4_ENV"     "$BD/c_onw4"
echo "banner_onjout=$(grep -c "$MARK" "$BD/c_onjout.err" || echo 0)  banner_onw2=$(grep -c "$MARK" "$BD/c_onw2.err" || echo 0)  banner_onw4=$(grep -c "$MARK" "$BD/c_onw4.err" || echo 0)"
echo "onw2_banner_line: $(grep -m1 "$MARK" "$BD/c_onw2.err" || echo NONE)"
echo "onw4_banner_line: $(grep -m1 "$MARK" "$BD/c_onw4.err" || echo NONE)"
echo "onw2_vs_onjout:  $( (cmp -s "$BD/c_onw2.out" "$BD/c_onjout.out" && echo IDENTICAL_widetile-neutral) || echo DIFFER_BUG )"
echo "onw4_vs_onjout:  $( (cmp -s "$BD/c_onw4.out" "$BD/c_onjout.out" && echo IDENTICAL_widetile-neutral) || echo DIFFER_BUG )"
echo "onw2_vs_off:     $( (cmp -s "$BD/c_onw2.out" "$BD/c_off.out" && echo IDENTICAL) || echo DIFFER )"
echo "onw4_vs_off:     $( (cmp -s "$BD/c_onw4.out" "$BD/c_off.out" && echo IDENTICAL) || echo DIFFER )"
echo "onjout_vs_off:   $( (cmp -s "$BD/c_onjout.out" "$BD/c_off.out" && echo IDENTICAL) || echo DIFFER )"
echo "md5_onjout=$(md5sum "$BD/c_onjout.out" | cut -d' ' -f1)  md5_onw2=$(md5sum "$BD/c_onw2.out" | cut -d' ' -f1)  md5_onw4=$(md5sum "$BD/c_onw4.out" | cut -d' ' -f1)  md5_off=$(md5sum "$BD/c_off.out" | cut -d' ' -f1)"
echo "seal_expect=$SEAL_MD5"
echo "----- OFF.out -----"; cat "$BD/c_off.out"; echo
echo "----- ONW2.out -----"; cat "$BD/c_onw2.out"; echo
echo "----- ONW4.out -----"; cat "$BD/c_onw4.out"; echo
echo "======== CORRECTNESS_END ========"

# ============================ [perf] interleaved paired ============================
echo "======== PERF_BEGIN ========"
run_side() { # $1=side  $2=bin  $3=envstr
  local side="$1" bin="$2" envs="$3" fk banner errf
  fk=$(freq); errf="$BD/bench_${side}_p${PASS}.err"
  echo "###AB pass=$PASS side=$side freq_khz=$fk"
  env $envs $PIN "$bin" -m "$MODEL" -p "$PP" -n 0 -t "$THREADS" -r "$REPS" -o json 2>"$errf"
  banner=$(grep -c "$MARK" "$errf" 2>/dev/null || echo 0)
  echo "###BANNER side=$side pass=$PASS count=$banner"
  echo "###END"
}
for PASS in $(seq 1 "$PASSES"); do
  echo "== ABPASS $PASS =="
  run_side off    "$BENCH_OFF" ""
  run_side ven    "$BENCH_IME" "-u $ENVV"
  run_side onjout "$BENCH_IME" "$NJOUT_ENV"
  run_side onw2   "$BENCH_IME" "$W2_ENV"
  run_side onw4   "$BENCH_IME" "$W4_ENV"
done
echo "======== PERF_END ========"
echo "loadavg_post=$(cat /proc/loadavg)"
echo "== DONE =="
# restore runs via EXIT trap
