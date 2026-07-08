#!/usr/bin/env bash
# g2_fuse_e2e_ab.sh -- G3 四.1 : whole-model e2e A/B for the rms_norm->mul FUSION,
# run ON the board. Phase-split (prefill pp + decode tg), paired-interleaved,
# cold(warmup-dropped) N>=10, T-N floor via PASSES, restore-clean.
#
# KEY FINDING that shapes this harness (see cell NOTES): upstream ggml (this build,
# f3e1828) ALREADY fuses RMS_NORM+MUL in its hand-written CPU path
# (ggml_compute_forward_rms_norm_mul_fused, ggml-cpu.c:3155), on by default,
# toggled ONLY by the env knob GGML_CPU_DISABLE_FUSION. So there is NO "unfused
# stock" build to compare against: the honest e2e A/B is fusion ON vs OFF on the
# SAME binary, which measures the exact transformation our compiler emits (the
# isolated cell proved our emitted kernel is bit-exact to ggml's fused kernel).
#
# Because A and B are literally the SAME .so with the same toolchain, preflight(0)
# toolchain symmetry is TOTAL by construction -- no compiler-asymmetry defect is
# possible (the only difference is one getenv() branch).
#
# Env: BENCH (llama-bench path) | MODEL | LDPATH | CORES (e.g. 8-15) | THREADS
#      PP TG REPS PASSES
# Emits ###FAB ... ###END blocks for the local aggregator.
set -u
BENCH="${BENCH:?set BENCH (llama-bench path)}"
MODEL="${MODEL:?set MODEL}"
CORES="${CORES:-8-15}"; THREADS="${THREADS:-8}"
PP="${PP:-128}"; TG="${TG:-32}"; REPS="${REPS:-5}"; PASSES="${PASSES:-2}"
[ -n "${LDPATH:-}" ] && export LD_LIBRARY_PATH="$LDPATH:${LD_LIBRARY_PATH:-}"
PIN="taskset -c $CORES"
CORE0=$(echo "$CORES" | grep -oE '^[0-9]+')
LIBDIR="$(dirname "$BENCH")"

echo "== G2 FUSION e2e A/B (fusion ON vs OFF, SAME binary) =="
echo "== PREFLIGHT(0): toolchain symmetry by construction =="
echo "bench=$BENCH"
echo "bench_sha256=$(sha256sum "$BENCH" 2>/dev/null | cut -c1-16)"
LIBCPU=$(ls "$LIBDIR"/libggml-cpu.so* 2>/dev/null | head -1)
echo "libggml-cpu=$LIBCPU  sha256=$(sha256sum "$LIBCPU" 2>/dev/null | cut -c1-16)"
KNOB=$(strings "$LIBCPU" 2>/dev/null | grep -c GGML_CPU_DISABLE_FUSION)
FUSED_SYM=$(strings "$LIBCPU" 2>/dev/null | grep -o rms_norm_mul_fused | head -1)
echo "fusion_knob_present=$KNOB  fused_symbol=$FUSED_SYM"
[ "$KNOB" -ge 1 ] || { echo "PREFLIGHT_FAIL: fusion knob absent in libggml-cpu; wrong build"; exit 30; }
[ -n "$FUSED_SYM" ] || { echo "PREFLIGHT_FAIL: rms_norm_mul_fused symbol absent; wrong build"; exit 30; }
MARCH=$(strings "$LIBCPU" 2>/dev/null | grep -oE 'rv64gcv[a-z0-9_]*' | head -1)
echo "march(baked)=$MARCH"

echo "== ENV FINGERPRINT (instance-hash inputs) =="
echo "uname=$(uname -a)"
echo "isa=$(grep -m1 -i isa /proc/cpuinfo)"
echo "nproc=$(nproc)  pinned_cores=$CORES  threads=$THREADS"
echo "governor=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_governor 2>/dev/null || echo NA)"
echo "cpu_max_khz=$(cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/cpuinfo_max_freq 2>/dev/null || echo NA)"
echo "model=$MODEL  model_sha256=$(sha256sum "$MODEL" 2>/dev/null | cut -c1-16)"
echo "config: PP=$PP TG=$TG REPS=$REPS PASSES=$PASSES"

freq() { cat /sys/devices/system/cpu/cpu${CORE0}/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

run_side() {  # $1=arm(on|off)  $2=pass
  local arm="$1" pass="$2" fk env=""
  fk=$(freq)
  [ "$arm" = "off" ] && env="GGML_CPU_DISABLE_FUSION=1"
  echo "###FAB pass=$pass arm=$arm freq_khz=$fk"
  env $env $PIN "$BENCH" -m "$MODEL" -p "$PP" -n "$TG" -t "$THREADS" -r "$REPS" -o json 2>/dev/null
  echo "###END"
}

echo "== WARMUP (dropped; pages in mmap, settles DVFS) =="
$PIN "$BENCH" -m "$MODEL" -p 16 -n 8 -t "$THREADS" -r 1 >/dev/null 2>&1
GGML_CPU_DISABLE_FUSION=1 $PIN "$BENCH" -m "$MODEL" -p 16 -n 8 -t "$THREADS" -r 1 >/dev/null 2>&1

for pass in $(seq 1 "$PASSES"); do
  echo "== FABPASS $pass =="
  run_side on  "$pass"
  run_side off "$pass"
done
echo "== DONE =="
