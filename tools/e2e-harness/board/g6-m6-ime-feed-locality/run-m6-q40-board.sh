#!/usr/bin/env bash
# G6-A M6 (B-feed locality) -- q4_0 IME bridge, BOARD-SIDE self-contained runner (k1).
# Single patch/build/restore cycle on top of the M5 (matmul-internal) bridge. Runs:
#   [X-0/Amdahl] compute-vs-feed split + loop-interchange decomposition (4-hart, ns_matmul =
#                parallel critical path, ith==0 timed). Five profiling configs (PARSETUP base):
#                  onepivec_full  = M5 mi-outer vec epilogue        -> t_M5 matmul
#                  onepivec_noepi = mi-outer vmadot only (skip fold) -> vmadot + B/A feed (mi-outer)
#                  compute_l1     = vmadot from FIXED L1 scratch     -> vmadot COMPUTE only
#                  onjout_full    = M6 nj-outer vec epilogue         -> t_M6 matmul
#                  onjout_noepi   = nj-outer vmadot only             -> vmadot + B/A feed (nj-outer)
#                feed_mi = noepi_mi - compute ; feed_njout = noepi_njout - compute
#                interchange feed saving = noepi_mi - noepi_njout ; M6 matmul saving = t_M5 - t_M6
#   [correctness] greedy A==B (HARD GATE): onjout vs onepivec MUST be byte-identical (loop
#                 interchange is pure tile-order reorder: each Cf[m,n] accumulates over b in the
#                 SAME order, single writer -> bit-exact). vs off = in-family coherence; vs ven =
#                 known near-tie flip. onepivec (M5 path in the M6 binary) vs off re-confirms
#                 M1..M5 continuity.
#   [perf] interleaved paired llama-bench, pp32 prefill, same -t:
#          off/ven/onepivec/onjout : every pass (>=12 samples/side)
#          off      = build-off              (stock RVV, no IME)
#          ven      = build-ime env unset    (vendor IME, deployed baseline)
#          onepivec = M5 (BRIDGE CACHE THREADS DEREF DEREF_EPI PARSETUP EPIVEC)
#          onjout   = M6 (+ NJOUTER : nj-outer B-feed locality)
# Our-N-hart vs vendor-N-hart matchup: ALL sides run -t $THREADS. decode (M=1) never routed.
# ALWAYS restores board (EXIT/INT/TERM + md5 double-proof).
set -uo pipefail
DIR="/home/bianbu/tcrv-k1-llama"
IME="$DIR/ggml/src/ggml-cpu/spacemit/ime.cpp"
MODEL="${MODEL:-$DIR/models/tinyllama-q4_0.gguf}"
BD="/tmp/g6m6q40"
BASE_IME="40962c7e7c732bf472ae88cef89ced8d"
BASE_SO="71cc4d295dac29382a0a7d4d5bd0c425"
BENCH_IME="$DIR/build-ime/bin/llama-bench"
BENCH_OFF="$DIR/build-off/bin/llama-bench"
CMPL_IME="$DIR/build-ime/bin/llama-completion"
CMPL_OFF="$DIR/build-off/bin/llama-completion"
PP="${PP:-32}"; REPS="${REPS:-4}"; PASSES="${PASSES:-3}"; THREADS="${THREADS:-4}"
CORES="${CORES:-0-3}"; CORE0=0
ENVV="TCRV_IME_Q40_BRIDGE"; CACHEV="TCRV_IME_Q40_CACHE"; THREADSV="TCRV_IME_Q40_THREADS"
DEREFV="TCRV_IME_Q40_DEREF"; DEREFEPIV="TCRV_IME_Q40_DEREF_EPI"; PARV="TCRV_IME_Q40_PARSETUP"
EPIVECV="TCRV_IME_Q40_EPIVEC"; NJOUTV="TCRV_IME_Q40_NJOUTER"
NOEPIV="TCRV_IME_Q40_MMPROF_NOEPI"; L1V="TCRV_IME_Q40_MMPROF_L1"; PROFV="TCRV_IME_Q40_PROF"
MARK="TCRV-IME-Q40-BRIDGE"
PIN="taskset -c $CORES"
EPIVEC_ENV="$ENVV=1 $CACHEV=1 $THREADSV=1 $DEREFV=1 $DEREFEPIV=1 $PARV=1 $EPIVECV=1"
NJOUT_ENV="$EPIVEC_ENV $NJOUTV=1"
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
  echo "ALL_DONE_q40_m6"
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
echo "config: PP=$PP REPS=$REPS PASSES=$PASSES  sides=off/ven/onepivec/onjout  prof=M5full/noepi_mi/compute_l1/M6full/noepi_njout"

echo "=== patch + rebuild build-ime (q4_0 B-feed-locality bridge) ==="
python3 "$BD/forward-route-patch-q40-feedloc.py"
if make -C build-ime ggml-cpu -j8 >"$BD/route_build.log" 2>&1; then echo "rebuild ok"; else echo "BUILD FAIL"; tail -80 "$BD/route_build.log"; exit 1; fi
echo "vmadot_in_patched_so=$(objdump -d "$SO" | grep -c vmadot || true)  (baseline 32; M5=38; M6 adds vec_njouter+noepi_njouter+noepi_l1 -> expect ~41)"

freq() { cat /sys/devices/system/cpu/cpu$CORE0/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

echo "== LOAD-GATE (static board check before measure) =="
echo "loadavg_pre=$(cat /proc/loadavg)"
echo "extern_jobs=$(pgrep -af 'llama-bench|llama-completion' | grep -v $$ | grep -vc pgrep || echo 0)"

echo "== WARMUP (dropped) =="
env $NJOUT_ENV $PIN "$BENCH_IME" -m "$MODEL" -p 16 -n 0 -t "$THREADS" -r 1 >/dev/null 2>&1 || true

# ============================ [X-0 / Amdahl] compute-vs-feed + interchange decomposition ============
echo "======== PROFILE_BEGIN ========"
echo "---- PROF onepivec_full (M5 mi-outer vec: vmadot+feed+vec-fold) ----"
env $EPIVEC_ENV "$PROFV=1" $PIN "$BENCH_IME" -m "$MODEL" -p "$PP" -n 0 -t "$THREADS" -r 3 -o json \
    >"$BD/prof_onepivec_full.json" 2>"$BD/prof_onepivec_full.err" || true
grep -E "TCRV-Q40-PROF|TCRV-IME-Q40-BRIDGE" "$BD/prof_onepivec_full.err" || echo "(no prof line onepivec_full!)"
echo "---- PROF onepivec_noepi (mi-outer vmadot only: vmadot + B/A feed) ----"
env $EPIVEC_ENV "$NOEPIV=1" "$PROFV=1" $PIN "$BENCH_IME" -m "$MODEL" -p "$PP" -n 0 -t "$THREADS" -r 3 -o json \
    >"$BD/prof_onepivec_noepi.json" 2>"$BD/prof_onepivec_noepi.err" || true
grep -E "TCRV-Q40-PROF|TCRV-IME-Q40-BRIDGE" "$BD/prof_onepivec_noepi.err" || echo "(no prof line onepivec_noepi!)"
echo "---- PROF compute_l1 (vmadot from FIXED L1 scratch: compute only) ----"
env $EPIVEC_ENV "$L1V=1" "$PROFV=1" $PIN "$BENCH_IME" -m "$MODEL" -p "$PP" -n 0 -t "$THREADS" -r 3 -o json \
    >"$BD/prof_compute_l1.json" 2>"$BD/prof_compute_l1.err" || true
grep -E "TCRV-Q40-PROF|TCRV-IME-Q40-BRIDGE" "$BD/prof_compute_l1.err" || echo "(no prof line compute_l1!)"
echo "---- PROF onjout_full (M6 nj-outer vec: vmadot+feed+vec-fold, B-locality) ----"
env $NJOUT_ENV "$PROFV=1" $PIN "$BENCH_IME" -m "$MODEL" -p "$PP" -n 0 -t "$THREADS" -r 3 -o json \
    >"$BD/prof_onjout_full.json" 2>"$BD/prof_onjout_full.err" || true
grep -E "TCRV-Q40-PROF|TCRV-IME-Q40-BRIDGE" "$BD/prof_onjout_full.err" || echo "(no prof line onjout_full!)"
echo "---- PROF onjout_noepi (nj-outer vmadot only: vmadot + B/A feed, B-locality) ----"
env $NJOUT_ENV "$NOEPIV=1" "$PROFV=1" $PIN "$BENCH_IME" -m "$MODEL" -p "$PP" -n 0 -t "$THREADS" -r 3 -o json \
    >"$BD/prof_onjout_noepi.json" 2>"$BD/prof_onjout_noepi.err" || true
grep -E "TCRV-Q40-PROF|TCRV-IME-Q40-BRIDGE" "$BD/prof_onjout_noepi.err" || echo "(no prof line onjout_noepi!)"
echo "======== PROFILE_END ========"

# ============================ [correctness] greedy A==B ============================
echo "======== CORRECTNESS_BEGIN ========"
CPROMPT="Once upon a time, in a small village nestled between two great mountains, there lived"
run_cmpl() { # $1=bin $2=env $3=stem
  env $2 $PIN "$1" -m "$MODEL" -p "$CPROMPT" -n 24 --temp 0 -s 0 -t "$THREADS" \
      -no-cnv --no-warmup --no-display-prompt --simple-io >"$3.out" 2>"$3.err" || true
}
run_cmpl "$CMPL_OFF" ""                "$BD/c_off"
run_cmpl "$CMPL_IME" ""                "$BD/c_ven"
run_cmpl "$CMPL_IME" "$EPIVEC_ENV"     "$BD/c_onepivec"
run_cmpl "$CMPL_IME" "$NJOUT_ENV"      "$BD/c_onjout"
echo "banner_onepivec=$(grep -c "$MARK" "$BD/c_onepivec.err" || echo 0)  banner_onjout=$(grep -c "$MARK" "$BD/c_onjout.err" || echo 0)  banner_ven=$(grep -c "$MARK" "$BD/c_ven.err" || echo 0)"
echo "onjout_banner_line: $(grep -m1 "$MARK" "$BD/c_onjout.err" || echo NONE)"
echo "onjout_vs_onepivec:  $( (cmp -s "$BD/c_onjout.out" "$BD/c_onepivec.out" && echo IDENTICAL_njouter-neutral) || echo DIFFER_BUG )"
echo "onjout_vs_off:       $( (cmp -s "$BD/c_onjout.out" "$BD/c_off.out"  && echo IDENTICAL) || echo DIFFER )"
echo "onepivec_vs_off:     $( (cmp -s "$BD/c_onepivec.out" "$BD/c_off.out" && echo IDENTICAL) || echo DIFFER )"
echo "onjout_vs_ven:       $( (cmp -s "$BD/c_onjout.out" "$BD/c_ven.out"  && echo IDENTICAL) || echo DIFFER )"
echo "md5_onjout=$(md5sum "$BD/c_onjout.out" | cut -d' ' -f1)  md5_onepivec=$(md5sum "$BD/c_onepivec.out" | cut -d' ' -f1)  md5_off=$(md5sum "$BD/c_off.out" | cut -d' ' -f1)"
echo "----- OFF.out -----"; cat "$BD/c_off.out"; echo
echo "----- ONJOUT.out -----"; cat "$BD/c_onjout.out"; echo
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
  run_side off      "$BENCH_OFF" ""
  run_side ven      "$BENCH_IME" "-u $ENVV"
  run_side onepivec "$BENCH_IME" "$EPIVEC_ENV"
  run_side onjout   "$BENCH_IME" "$NJOUT_ENV"
done
echo "======== PERF_END ========"
echo "loadavg_post=$(cat /proc/loadavg)"
echo "== DONE =="
# restore runs via EXIT trap
