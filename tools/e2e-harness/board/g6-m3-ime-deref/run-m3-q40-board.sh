#!/usr/bin/env bash
# G6-A M3 (de-reference-form) -- q4_0 IME bridge, BOARD-SIDE self-contained runner (k1).
# Single patch/build/restore cycle on top of the M2 (multithread) bridge. Runs:
#   [X-0/Amdahl] profiling (4-hart, ns_matmul = parallel critical path, ith==0 timed):
#                onmt       = M2 baseline (per-call dequant + per-block Cf accumulate)
#                onderef_dq = M3-a       (cached dequant B  + per-block Cf accumulate)  -> dequant saving
#                onderef    = M3 full    (cached dequant B  + register-acc epilogue)     -> +epilogue saving
#                dequant saving  = onmt.ns_matmul  - onderef_dq.ns_matmul
#                epilogue saving = onderef_dq.ns_matmul - onderef.ns_matmul
#   [correctness] greedy A==B (HARD GATE): onderef vs onmt MUST be byte-identical
#                 (de-reference-form is numerically neutral: same vmadot bytes, same scalar
#                 f32 expression, same summation order). onderef_dq vs onmt identical too.
#                 vs off = in-family coherence (MIRAGE excluded); vs ven = known near-tie flip.
#   [perf] interleaved paired llama-bench, pp32 prefill, same -t:
#          off/ven/onmt/onderef  : every pass (>=12 samples/side)
#          onnc/oncache          : pass-1 only (4-sample continuity anchors; 12-sample values
#                                  are sealed in M1/M2 -> harness fidelity spot-check here)
#          off     = build-off              (stock RVV, no IME)
#          ven     = build-ime env unset    (vendor IME, deployed baseline)
#          onnc    = BRIDGE=1                          (per-call repack, single hart = sealed base)
#          oncache = BRIDGE=1 CACHE=1                  (M1 cached repack, single hart)
#          onmt    = BRIDGE=1 CACHE=1 THREADS=1        (M2 cached repack + column-tile multithread)
#          onderef = BRIDGE=1 CACHE=1 THREADS=1 DEREF=1 DEREF_EPI=1  (M3 = + cached dequant B + reg epilogue)
# Our-N-hart vs vendor-N-hart matchup: ALL sides run -t $THREADS. decode (M=1) is NEVER routed
# (M>1 gate) -> M1-neutral by construction. ALWAYS restores board (EXIT/INT/TERM + md5 double-proof).
set -uo pipefail
DIR="/home/bianbu/tcrv-k1-llama"
IME="$DIR/ggml/src/ggml-cpu/spacemit/ime.cpp"
MODEL="${MODEL:-$DIR/models/tinyllama-q4_0.gguf}"
BD="/tmp/g6m3q40"
BASE_IME="40962c7e7c732bf472ae88cef89ced8d"
BASE_SO="71cc4d295dac29382a0a7d4d5bd0c425"
BENCH_IME="$DIR/build-ime/bin/llama-bench"
BENCH_OFF="$DIR/build-off/bin/llama-bench"
CMPL_IME="$DIR/build-ime/bin/llama-completion"
CMPL_OFF="$DIR/build-off/bin/llama-completion"
PP="${PP:-32}"; REPS="${REPS:-4}"; PASSES="${PASSES:-3}"; THREADS="${THREADS:-4}"
CORES="${CORES:-0-3}"; CORE0=0
ENVV="TCRV_IME_Q40_BRIDGE"; CACHEV="TCRV_IME_Q40_CACHE"; THREADSV="TCRV_IME_Q40_THREADS"
DEREFV="TCRV_IME_Q40_DEREF"; DEREFEPIV="TCRV_IME_Q40_DEREF_EPI"; PROFV="TCRV_IME_Q40_PROF"
MARK="TCRV-IME-Q40-BRIDGE"
PIN="taskset -c $CORES"
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
  echo "ALL_DONE_q40_m3"
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
echo "config: PP=$PP REPS=$REPS PASSES=$PASSES  sides=off/ven/onnc/oncache/onmt/onderef"

echo "=== patch + rebuild build-ime (q4_0 de-reference-form bridge) ==="
python3 "$BD/forward-route-patch-q40-deref.py"
if make -C build-ime ggml-cpu -j8 >"$BD/route_build.log" 2>&1; then echo "rebuild ok"; else echo "BUILD FAIL"; tail -60 "$BD/route_build.log"; exit 1; fi
echo "vmadot_in_patched_so=$(objdump -d "$SO" | grep -c vmadot || true)  (baseline 32; M2=34; M3 adds 2 deref matmuls -> expect ~36)"

freq() { cat /sys/devices/system/cpu/cpu$CORE0/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

echo "== WARMUP (dropped) =="
env "$ENVV=1" "$CACHEV=1" "$THREADSV=1" "$DEREFV=1" "$DEREFEPIV=1" $PIN "$BENCH_IME" -m "$MODEL" -p 16 -n 0 -t "$THREADS" -r 1 >/dev/null 2>&1 || true

# ============================ [X-0 / Amdahl] profiling (4-hart, ns_matmul internal split) ========
echo "======== PROFILE_BEGIN ========"
echo "---- PROF onmt (BRIDGE=1 CACHE=1 THREADS=1 PROF=1: M2 per-call dequant + per-block Cf) ----"
env "$ENVV=1" "$CACHEV=1" "$THREADSV=1" "$PROFV=1" $PIN "$BENCH_IME" -m "$MODEL" -p "$PP" -n 0 -t "$THREADS" -r 3 -o json \
    >"$BD/prof_onmt.json" 2>"$BD/prof_onmt.err" || true
grep -E "TCRV-Q40-PROF|TCRV-IME-Q40-BRIDGE" "$BD/prof_onmt.err" || echo "(no prof line onmt!)"
echo "---- PROF onderef_dq (BRIDGE=1 THREADS=1 DEREF=1 PROF=1: cached dequant B + per-block Cf) ----"
env "$ENVV=1" "$THREADSV=1" "$DEREFV=1" "$PROFV=1" $PIN "$BENCH_IME" -m "$MODEL" -p "$PP" -n 0 -t "$THREADS" -r 3 -o json \
    >"$BD/prof_onderef_dq.json" 2>"$BD/prof_onderef_dq.err" || true
grep -E "TCRV-Q40-PROF|TCRV-IME-Q40-BRIDGE" "$BD/prof_onderef_dq.err" || echo "(no prof line onderef_dq!)"
echo "---- PROF onderef (BRIDGE=1 THREADS=1 DEREF=1 DEREF_EPI=1 PROF=1: cached dequant B + reg epilogue) ----"
env "$ENVV=1" "$THREADSV=1" "$DEREFV=1" "$DEREFEPIV=1" "$PROFV=1" $PIN "$BENCH_IME" -m "$MODEL" -p "$PP" -n 0 -t "$THREADS" -r 3 -o json \
    >"$BD/prof_onderef.json" 2>"$BD/prof_onderef.err" || true
grep -E "TCRV-Q40-PROF|TCRV-IME-Q40-BRIDGE" "$BD/prof_onderef.err" || echo "(no prof line onderef!)"
echo "======== PROFILE_END ========"

# ============================ [correctness] greedy A==B ============================
echo "======== CORRECTNESS_BEGIN ========"
CPROMPT="Once upon a time, in a small village nestled between two great mountains, there lived"
run_cmpl() { # $1=bin $2=env $3=stem
  env $2 $PIN "$1" -m "$MODEL" -p "$CPROMPT" -n 24 --temp 0 -s 0 -t "$THREADS" \
      -no-cnv --no-warmup --no-display-prompt --simple-io >"$3.out" 2>"$3.err" || true
}
run_cmpl "$CMPL_OFF" ""                                                          "$BD/c_off"
run_cmpl "$CMPL_IME" ""                                                          "$BD/c_ven"
run_cmpl "$CMPL_IME" "$ENVV=1"                                                   "$BD/c_onnc"
run_cmpl "$CMPL_IME" "$ENVV=1 $CACHEV=1"                                         "$BD/c_oncache"
run_cmpl "$CMPL_IME" "$ENVV=1 $CACHEV=1 $THREADSV=1"                             "$BD/c_onmt"
run_cmpl "$CMPL_IME" "$ENVV=1 $THREADSV=1 $DEREFV=1"                             "$BD/c_onderef_dq"
run_cmpl "$CMPL_IME" "$ENVV=1 $CACHEV=1 $THREADSV=1 $DEREFV=1 $DEREFEPIV=1"      "$BD/c_onderef"
echo "banner_onnc=$(grep -c "$MARK" "$BD/c_onnc.err" || echo 0)  banner_oncache=$(grep -c "$MARK" "$BD/c_oncache.err" || echo 0)  banner_onmt=$(grep -c "$MARK" "$BD/c_onmt.err" || echo 0)  banner_onderef_dq=$(grep -c "$MARK" "$BD/c_onderef_dq.err" || echo 0)  banner_onderef=$(grep -c "$MARK" "$BD/c_onderef.err" || echo 0)  banner_ven=$(grep -c "$MARK" "$BD/c_ven.err" || echo 0)"
echo "onderef_banner_line: $(grep -m1 "$MARK" "$BD/c_onderef.err" || echo NONE)"
echo "onderef_vs_onmt:     $( (cmp -s "$BD/c_onderef.out" "$BD/c_onmt.out" && echo IDENTICAL_deref-neutral) || echo DIFFER_BUG )"
echo "onderef_dq_vs_onmt:  $( (cmp -s "$BD/c_onderef_dq.out" "$BD/c_onmt.out" && echo IDENTICAL_dq-neutral) || echo DIFFER_BUG )"
echo "onderef_vs_oncache:  $( (cmp -s "$BD/c_onderef.out" "$BD/c_oncache.out" && echo IDENTICAL) || echo DIFFER )"
echo "onmt_vs_oncache:     $( (cmp -s "$BD/c_onmt.out" "$BD/c_oncache.out" && echo IDENTICAL) || echo DIFFER )"
echo "oncache_vs_onnc:     $( (cmp -s "$BD/c_oncache.out" "$BD/c_onnc.out" && echo IDENTICAL) || echo DIFFER )"
echo "onderef_vs_off:      $( (cmp -s "$BD/c_onderef.out" "$BD/c_off.out"  && echo IDENTICAL) || echo DIFFER )"
echo "onderef_vs_ven:      $( (cmp -s "$BD/c_onderef.out" "$BD/c_ven.out"  && echo IDENTICAL) || echo DIFFER )"
echo "----- OFF.out -----"; cat "$BD/c_off.out"; echo
echo "----- ONDEREF.out -----"; cat "$BD/c_onderef.out"; echo
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
  run_side off     "$BENCH_OFF" ""
  run_side ven     "$BENCH_IME" "-u $ENVV"
  if [ "$PASS" = "1" ]; then
    run_side onnc    "$BENCH_IME" "$ENVV=1"
    run_side oncache "$BENCH_IME" "$ENVV=1 $CACHEV=1"
  fi
  run_side onmt    "$BENCH_IME" "$ENVV=1 $CACHEV=1 $THREADSV=1"
  run_side onderef "$BENCH_IME" "$ENVV=1 $CACHEV=1 $THREADSV=1 $DEREFV=1 $DEREFEPIV=1"
done
echo "======== PERF_END ========"
echo "== DONE =="
# restore runs via EXIT trap
