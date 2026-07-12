#!/usr/bin/env bash
# G5-M3 IME e2e ON/OFF PAIRED llama-bench (perf) -- BOARD-SIDE self-contained runner.
# Runs ENTIRELY on k1, launched under nohup by run-pair-bg.sh (survives ssh
# disconnect; polled via logfile for ALL_DONE_<fmt>).  Args:
#   $1=fmt  $2=model  $3=patch_basename  $4=envvar  $5=marker  [$6=PP_LIST] [$7=TG] [$8=REPS] [$9=PASSES]
#
# THREE interleaved sides (drift-cancel), toggling ONLY the routing:
#   ON  = build-ime env $ENVV=1   -> our tcrv IME kernel routes real PREFILL mul_mat (banner fires)
#   VEN = build-ime env -u $ENVV   -> vendor IME path (deployed baseline for IME buffers; byte-identical A/B toggle)
#   OFF = build-off llama-bench    -> stock RVV vec_dot (no IME) = the RVV-vector base of the 2.09x compute-account
# `-p PP` = prefill (M>1 -> IME array); `-n TG` = decode (M=1 -> native fallback, control).
# Patch vendor ime.cpp with the format's env-gated bridge, rebuild build-ime ONCE.
# ALWAYS restores board to byte-identical baseline (EXIT/INT/TERM trap + md5 double-check).
set -uo pipefail
FMT="$1"; MODEL="$2"; PATCH_BN="$3"; ENVV="$4"; MARK="$5"
PP_LIST="${6:-32 256}"; TG="${7:-64}"; REPS="${8:-5}"; PASSES="${9:-2}"; THREADS="${THREADS:-4}"
CORES="${CORES:-0-3}"; CORE0=0
DIR="/home/bianbu/tcrv-k1-llama"
IME="$DIR/ggml/src/ggml-cpu/spacemit/ime.cpp"
BD="/tmp/g5m3pair/$FMT"
BASE_IME="40962c7e7c732bf472ae88cef89ced8d"
BASE_SO="71cc4d295dac29382a0a7d4d5bd0c425"
BENCH_IME="$DIR/build-ime/bin/llama-bench"
BENCH_OFF="$DIR/build-off/bin/llama-bench"
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
  echo "ALL_DONE_$FMT"
}
trap restore EXIT INT TERM

PIN="taskset -c $CORES"
PPARGS=""; for p in $PP_LIST; do PPARGS="$PPARGS -p $p"; done

echo "== ENV FINGERPRINT =="
echo "uname=$(uname -a)"
echo "isa=$(grep -m1 -i isa /proc/cpuinfo)"
echo "nproc=$(nproc) pinned_cores=$CORES threads=$THREADS"
echo "governor=$(cat /sys/devices/system/cpu/cpu$CORE0/cpufreq/scaling_governor 2>/dev/null || echo NA)"
echo "cpu_max_khz=$(cat /sys/devices/system/cpu/cpu$CORE0/cpufreq/cpuinfo_max_freq 2>/dev/null || echo NA)"
echo "clang: $(clang --version 2>/dev/null | head -1)  (kernel==system: shipped clang-18 both builds)"
echo "model=$MODEL model_sha256=$(sha256sum "$MODEL" 2>/dev/null | cut -c1-16)"
echo "bench_ime=$BENCH_IME  bench_off=$BENCH_OFF"
echo "config: PP_LIST='$PP_LIST' TG=$TG REPS=$REPS PASSES=$PASSES  sides=on(tcrv-IME)/ven(vendor-IME)/off(stock-RVV)"

echo "=== patch + rebuild build-ime ($FMT bridge) ==="
python3 "$BD/$PATCH_BN"
if make -C build-ime ggml-cpu -j8 >"$BD/route_build.log" 2>&1; then echo "rebuild ok"; else echo "BUILD FAIL"; tail -40 "$BD/route_build.log"; exit 1; fi
echo "vmadot_in_patched_so=$(objdump -d "$SO" | grep -c vmadot || true)  (baseline 32; >32 = tcrv IME kernel present)"

# ---- board-load gate: informational (cores 0-3 are our pinned target; consumers live on 5-7).
#      interleaved A/B cancels symmetric background load. record loadavg, brief settle wait. ----
echo "== BOARD-LOAD GATE (informational) =="
for i in $(seq 1 6); do
  LA=$(cut -d' ' -f1 /proc/loadavg)
  echo "  loadavg1=$LA (try $i/6)"
  ok=$(awk -v l="$LA" 'BEGIN{print (l<2.5)?1:0}')
  [ "$ok" = "1" ] && break
  sleep 5
done
echo "board_load_gate_final_loadavg1=$(cut -d' ' -f1 /proc/loadavg)"

freq() { cat /sys/devices/system/cpu/cpu$CORE0/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

echo "== WARMUP (dropped, all sides) =="
env "$ENVV=1" $PIN "$BENCH_IME" -m "$MODEL" -p 16 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1
env -u "$ENVV" $PIN "$BENCH_IME" -m "$MODEL" -p 16 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1
$PIN "$BENCH_OFF" -m "$MODEL" -p 16 -n 4 -t "$THREADS" -r 1 >/dev/null 2>&1

run_side() {  # $1=side(on|ven|off)
  local side="$1" fk banner errf
  fk=$(freq)
  errf="$BD/bench_${side}_p${PASS}.err"
  echo "###AB pass=$PASS side=$side freq_khz=$fk"
  case "$side" in
    on)  env "$ENVV=1" $PIN "$BENCH_IME" -m "$MODEL" $PPARGS -n "$TG" -t "$THREADS" -r "$REPS" -o json 2>"$errf" ;;
    ven) env -u "$ENVV" $PIN "$BENCH_IME" -m "$MODEL" $PPARGS -n "$TG" -t "$THREADS" -r "$REPS" -o json 2>"$errf" ;;
    off) $PIN "$BENCH_OFF" -m "$MODEL" $PPARGS -n "$TG" -t "$THREADS" -r "$REPS" -o json 2>"$errf" ;;
  esac
  banner=$(grep -c "$MARK" "$errf" 2>/dev/null || echo 0)
  echo "###BANNER side=$side pass=$PASS count=$banner"
  echo "###END"
}

for PASS in $(seq 1 "$PASSES"); do
  echo "== ABPASS $PASS =="
  run_side on
  run_side ven
  run_side off
done
echo "== DONE =="
# restore runs via EXIT trap
