#!/usr/bin/env bash
# G6-A M5 (matmul-internal) -- q4_0 IME bridge, BOARD-SIDE self-contained runner (k1).
# Single patch/build/restore cycle on top of the M4 (parallel-setup) bridge. Runs:
#   [X-0/Amdahl] matmul-internal decomposition (4-hart, ns_matmul = parallel critical path,
#                ith==0 timed). Four profiling configs (all PARSETUP base = M4 setup):
#                  onpar_full  = M4 scalar epilogue            -> ns_matmul = vmadot+feed+scalar-fold
#                  onpar_noepi = vmadot only (skip fold)       -> ns_matmul ~= vmadot + B/A feed
#                  onpar_nomad = fold only (skip vmadot)       -> ns_matmul ~= scalar epilogue
#                  onepivec    = M5 vectorized epilogue        -> ns_matmul = vmadot+feed+vec-fold
#                epilogue-marginal = onpar_full - onpar_noepi ; vmadot+feed = onpar_noepi
#                M5 matmul saving = onpar_full - onepivec
#   [correctness] greedy A==B (HARD GATE): onepivec vs onpar MUST be byte-identical (the vec fold
#                 is a bit-exact SIMD reshape: per lane vfmul(dA,dW)+vfcvt(frag)+vfmacc == scalar
#                 fmul+fmadd, same b-order per independent column). vs off = in-family coherence
#                 (MIRAGE excluded); vs ven = known near-tie flip. onpar (M4 path in the M5 binary)
#                 vs off re-confirms M1..M4 continuity.
#   [perf] interleaved paired llama-bench, pp32 prefill, same -t:
#          off/ven/onpar/onepivec : every pass (>=12 samples/side)
#          off      = build-off              (stock RVV, no IME)
#          ven      = build-ime env unset    (vendor IME, deployed baseline)
#          onpar    = M4 (BRIDGE CACHE THREADS DEREF DEREF_EPI PARSETUP)
#          onepivec = M5 (+ EPIVEC : vectorized epilogue)
# Our-N-hart vs vendor-N-hart matchup: ALL sides run -t $THREADS. decode (M=1) never routed.
# ALWAYS restores board (EXIT/INT/TERM + md5 double-proof).
set -uo pipefail
DIR="/home/bianbu/tcrv-k1-llama"
IME="$DIR/ggml/src/ggml-cpu/spacemit/ime.cpp"
MODEL="${MODEL:-$DIR/models/tinyllama-q4_0.gguf}"
BD="/tmp/g6m5q40"
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
EPIVECV="TCRV_IME_Q40_EPIVEC"; NOEPIV="TCRV_IME_Q40_MMPROF_NOEPI"; NOMADV="TCRV_IME_Q40_MMPROF_NOMADOT"
PROFV="TCRV_IME_Q40_PROF"
MARK="TCRV-IME-Q40-BRIDGE"
PIN="taskset -c $CORES"
PAR_ENV="$ENVV=1 $CACHEV=1 $THREADSV=1 $DEREFV=1 $DEREFEPIV=1 $PARV=1"
EPIVEC_ENV="$PAR_ENV $EPIVECV=1"
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
  echo "ALL_DONE_q40_m5"
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
echo "config: PP=$PP REPS=$REPS PASSES=$PASSES  sides=off/ven/onpar/onepivec  prof=full/noepi/nomadot/epivec"

echo "=== patch + rebuild build-ime (q4_0 matmul-internal bridge) ==="
python3 "$BD/forward-route-patch-q40-matmul.py"
if make -C build-ime ggml-cpu -j8 >"$BD/route_build.log" 2>&1; then echo "rebuild ok"; else echo "BUILD FAIL"; tail -80 "$BD/route_build.log"; exit 1; fi
echo "vmadot_in_patched_so=$(objdump -d "$SO" | grep -c vmadot || true)  (baseline 32; M3=36; M5 adds vec+noepi matmuls -> expect ~38)"

freq() { cat /sys/devices/system/cpu/cpu$CORE0/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

echo "== LOAD-GATE (static board check before measure) =="
echo "loadavg_pre=$(cat /proc/loadavg)"
echo "extern_jobs=$(pgrep -af 'llama-bench|llama-completion' | grep -v $$ | grep -vc pgrep || echo 0)"

echo "== WARMUP (dropped) =="
env $EPIVEC_ENV $PIN "$BENCH_IME" -m "$MODEL" -p 16 -n 0 -t "$THREADS" -r 1 >/dev/null 2>&1 || true

# ============================ [X-0 / Amdahl] matmul-internal decomposition ================
echo "======== PROFILE_BEGIN ========"
echo "---- PROF onpar_full (M4 scalar epilogue: vmadot+feed+fold) ----"
env $PAR_ENV "$PROFV=1" $PIN "$BENCH_IME" -m "$MODEL" -p "$PP" -n 0 -t "$THREADS" -r 3 -o json \
    >"$BD/prof_onpar_full.json" 2>"$BD/prof_onpar_full.err" || true
grep -E "TCRV-Q40-PROF|TCRV-IME-Q40-BRIDGE" "$BD/prof_onpar_full.err" || echo "(no prof line onpar_full!)"
echo "---- PROF onpar_noepi (vmadot only, skip fold: vmadot + B/A feed) ----"
env $PAR_ENV "$NOEPIV=1" "$PROFV=1" $PIN "$BENCH_IME" -m "$MODEL" -p "$PP" -n 0 -t "$THREADS" -r 3 -o json \
    >"$BD/prof_onpar_noepi.json" 2>"$BD/prof_onpar_noepi.err" || true
grep -E "TCRV-Q40-PROF|TCRV-IME-Q40-BRIDGE" "$BD/prof_onpar_noepi.err" || echo "(no prof line onpar_noepi!)"
echo "---- PROF onpar_nomad (fold only, skip vmadot: scalar epilogue) ----"
env $PAR_ENV "$NOMADV=1" "$PROFV=1" $PIN "$BENCH_IME" -m "$MODEL" -p "$PP" -n 0 -t "$THREADS" -r 3 -o json \
    >"$BD/prof_onpar_nomad.json" 2>"$BD/prof_onpar_nomad.err" || true
grep -E "TCRV-Q40-PROF|TCRV-IME-Q40-BRIDGE" "$BD/prof_onpar_nomad.err" || echo "(no prof line onpar_nomad!)"
echo "---- PROF onepivec (M5 vectorized epilogue: vmadot+feed+vec-fold) ----"
env $EPIVEC_ENV "$PROFV=1" $PIN "$BENCH_IME" -m "$MODEL" -p "$PP" -n 0 -t "$THREADS" -r 3 -o json \
    >"$BD/prof_onepivec.json" 2>"$BD/prof_onepivec.err" || true
grep -E "TCRV-Q40-PROF|TCRV-IME-Q40-BRIDGE" "$BD/prof_onepivec.err" || echo "(no prof line onepivec!)"
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
run_cmpl "$CMPL_IME" "$PAR_ENV"        "$BD/c_onpar"
run_cmpl "$CMPL_IME" "$EPIVEC_ENV"     "$BD/c_onepivec"
echo "banner_onpar=$(grep -c "$MARK" "$BD/c_onpar.err" || echo 0)  banner_onepivec=$(grep -c "$MARK" "$BD/c_onepivec.err" || echo 0)  banner_ven=$(grep -c "$MARK" "$BD/c_ven.err" || echo 0)"
echo "onepivec_banner_line: $(grep -m1 "$MARK" "$BD/c_onepivec.err" || echo NONE)"
echo "onepivec_vs_onpar:   $( (cmp -s "$BD/c_onepivec.out" "$BD/c_onpar.out" && echo IDENTICAL_epivec-neutral) || echo DIFFER_BUG )"
echo "onepivec_vs_off:     $( (cmp -s "$BD/c_onepivec.out" "$BD/c_off.out"  && echo IDENTICAL) || echo DIFFER )"
echo "onpar_vs_off:        $( (cmp -s "$BD/c_onpar.out"    "$BD/c_off.out"  && echo IDENTICAL) || echo DIFFER )"
echo "onepivec_vs_ven:     $( (cmp -s "$BD/c_onepivec.out" "$BD/c_ven.out"  && echo IDENTICAL) || echo DIFFER )"
echo "md5_onepivec=$(md5sum "$BD/c_onepivec.out" | cut -d' ' -f1)  md5_onpar=$(md5sum "$BD/c_onpar.out" | cut -d' ' -f1)  md5_off=$(md5sum "$BD/c_off.out" | cut -d' ' -f1)"
echo "----- OFF.out -----"; cat "$BD/c_off.out"; echo
echo "----- ONEPIVEC.out -----"; cat "$BD/c_onepivec.out"; echo
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
  run_side onpar    "$BENCH_IME" "$PAR_ENV"
  run_side onepivec "$BENCH_IME" "$EPIVEC_ENV"
done
echo "======== PERF_END ========"
echo "loadavg_post=$(cat /proc/loadavg)"
echo "== DONE =="
# restore runs via EXIT trap
