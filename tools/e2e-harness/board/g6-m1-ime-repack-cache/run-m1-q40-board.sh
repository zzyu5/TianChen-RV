#!/usr/bin/env bash
# G6-A M1 (repack caching) -- q4_0 IME bridge, BOARD-SIDE self-contained runner (k1).
# Single patch/build/restore cycle. Runs, in order:
#   [X-0/Amdahl] profiling: per-stage ns split (repack/quant/matmul) cache OFF vs ON.
#   [correctness] greedy A==B: oncache vs onnc MUST be byte-identical (cache = same
#                 repack result, numerically neutral); vs ven/off = in-family coherence.
#   [perf] 4-side interleaved paired llama-bench, pp32 prefill, 12 samples/side:
#          onnc  = BRIDGE=1            (sealed session-3 bridge, per-call repack = OFF baseline)
#          oncache = BRIDGE=1 CACHE=1  (M1 = load-once cached repack)
#          ven   = build-ime env unset (vendor IME, deployed baseline)
#          off   = build-off           (stock RVV, no IME)
# decode (M=1) is NEVER routed (M>1 gate) and the env-gated native-passthrough layout is
# identical for cache on/off -> decode is M1-neutral by construction (not re-measured;
# sealed pairing tg64 ON/OFF=0.715x stands). ALWAYS restores board (EXIT/INT/TERM + md5).
set -uo pipefail
DIR="/home/bianbu/tcrv-k1-llama"
IME="$DIR/ggml/src/ggml-cpu/spacemit/ime.cpp"
MODEL="${MODEL:-$DIR/models/tinyllama-q4_0.gguf}"
BD="/tmp/g6m1q40"
BASE_IME="40962c7e7c732bf472ae88cef89ced8d"
BASE_SO="71cc4d295dac29382a0a7d4d5bd0c425"
BENCH_IME="$DIR/build-ime/bin/llama-bench"
BENCH_OFF="$DIR/build-off/bin/llama-bench"
CMPL_IME="$DIR/build-ime/bin/llama-completion"
CMPL_OFF="$DIR/build-off/bin/llama-completion"
PP="${PP:-32}"; REPS="${REPS:-4}"; PASSES="${PASSES:-3}"; THREADS="${THREADS:-4}"
CORES="${CORES:-0-3}"; CORE0=0
ENVV="TCRV_IME_Q40_BRIDGE"; CACHEV="TCRV_IME_Q40_CACHE"; PROFV="TCRV_IME_Q40_PROF"
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
  echo "ALL_DONE_q40_m1"
}
trap restore EXIT INT TERM

echo "== ENV FINGERPRINT =="
echo "uname=$(uname -a)"
echo "isa=$(grep -m1 -i isa /proc/cpuinfo)"
echo "nproc=$(nproc) pinned_cores=$CORES threads=$THREADS"
echo "governor=$(cat /sys/devices/system/cpu/cpu$CORE0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
echo "cur_freq=$(cat /sys/devices/system/cpu/cpu$CORE0/cpufreq/scaling_cur_freq 2>/dev/null || echo NA)"
echo "clang: $(clang --version 2>/dev/null | head -1)  (kernel==system: shipped clang-18 both builds)"
echo "model=$MODEL model_sha256=$(sha256sum "$MODEL" 2>/dev/null | cut -c1-16)"
echo "config: PP=$PP REPS=$REPS PASSES=$PASSES  sides=onnc/oncache/ven/off"

echo "=== patch + rebuild build-ime (q4_0 cache bridge) ==="
python3 "$BD/forward-route-patch-q40-cache.py"
if make -C build-ime ggml-cpu -j8 >"$BD/route_build.log" 2>&1; then echo "rebuild ok"; else echo "BUILD FAIL"; tail -50 "$BD/route_build.log"; exit 1; fi
echo "vmadot_in_patched_so=$(objdump -d "$SO" | grep -c vmadot || true)  (baseline 32; >32 = tcrv IME kernel present)"

freq() { cat /sys/devices/system/cpu/cpu$CORE0/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

echo "== WARMUP (dropped) =="
env "$ENVV=1" $PIN "$BENCH_IME" -m "$MODEL" -p 16 -n 0 -t "$THREADS" -r 1 >/dev/null 2>&1 || true

# ============================ [X-0 / Amdahl] profiling ============================
echo "======== PROFILE_BEGIN ========"
echo "---- PROF nocache (BRIDGE=1 PROF=1, cache off: per-call repack -> repack share of kernel) ----"
env "$ENVV=1" "$PROFV=1" $PIN "$BENCH_IME" -m "$MODEL" -p "$PP" -n 0 -t "$THREADS" -r 3 -o json \
    >"$BD/prof_nocache.json" 2>"$BD/prof_nocache.err" || true
grep -E "TCRV-Q40-PROF|TCRV-IME-Q40-BRIDGE" "$BD/prof_nocache.err" || echo "(no prof line nocache!)"
echo "---- PROF cache (BRIDGE=1 CACHE=1 PROF=1: repack_runs small, cache_hits large) ----"
env "$ENVV=1" "$CACHEV=1" "$PROFV=1" $PIN "$BENCH_IME" -m "$MODEL" -p "$PP" -n 0 -t "$THREADS" -r 3 -o json \
    >"$BD/prof_cache.json" 2>"$BD/prof_cache.err" || true
grep -E "TCRV-Q40-PROF|TCRV-IME-Q40-BRIDGE" "$BD/prof_cache.err" || echo "(no prof line cache!)"
echo "======== PROFILE_END ========"

# ============================ [correctness] greedy A==B ============================
echo "======== CORRECTNESS_BEGIN ========"
CPROMPT="Once upon a time, in a small village nestled between two great mountains, there lived"
run_cmpl() { # $1=bin $2=env $3=stem
  env $2 $PIN "$1" -m "$MODEL" -p "$CPROMPT" -n 24 --temp 0 -s 0 -t "$THREADS" \
      -no-cnv --no-warmup --no-display-prompt --simple-io >"$3.out" 2>"$3.err" || true
}
run_cmpl "$CMPL_OFF" ""                                "$BD/c_off"
run_cmpl "$CMPL_IME" ""                                "$BD/c_ven"
run_cmpl "$CMPL_IME" "$ENVV=1"                         "$BD/c_onnc"
run_cmpl "$CMPL_IME" "$ENVV=1 $CACHEV=1"               "$BD/c_oncache"
echo "banner_onnc=$(grep -c "$MARK" "$BD/c_onnc.err" || echo 0)  banner_oncache=$(grep -c "$MARK" "$BD/c_oncache.err" || echo 0)  banner_ven=$(grep -c "$MARK" "$BD/c_ven.err" || echo 0)"
echo "oncache_vs_onnc:  $( (cmp -s "$BD/c_oncache.out" "$BD/c_onnc.out" && echo IDENTICAL_cache-neutral) || echo DIFFER_BUG )"
echo "oncache_vs_ven:   $( (cmp -s "$BD/c_oncache.out" "$BD/c_ven.out"  && echo IDENTICAL) || echo DIFFER )"
echo "oncache_vs_off:   $( (cmp -s "$BD/c_oncache.out" "$BD/c_off.out"  && echo IDENTICAL) || echo DIFFER )"
echo "----- OFF.out -----"; cat "$BD/c_off.out"; echo
echo "----- ONCACHE.out -----"; cat "$BD/c_oncache.out"; echo
echo "======== CORRECTNESS_END ========"

# ============================ [perf] 4-side interleaved paired ============================
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
  run_side onnc    "$BENCH_IME" "$ENVV=1"
  run_side oncache "$BENCH_IME" "$ENVV=1 $CACHEV=1"
done
echo "======== PERF_END ========"
echo "== DONE =="
# restore runs via EXIT trap
