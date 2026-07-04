#!/usr/bin/env bash
# board_ab.sh -- constitutional paired A/B for q8_0 vec_dot micro-fixed.
# Runs ON the board. Compiles our kernel + faithful ggml factory with the SAME
# board clang / same flags (fp16 path cancels), pins to one core, interleaves
# our/factory, prints raw per-invocation kernel-ns. Two full passes => T-N floor.
#
# args: $1=RDIR  $2=OUR_KERNEL_CPP_basename  $3=CORE  $4=LABEL(m1|m2)  $5=DO_WINA(1|0)
set -u
RDIR="$1"; OURK="$2"; CORE="${3:-4}"; LABEL="${4:-mX}"; DO_WINA="${5:-0}"
cd "$RDIR"
C=$(command -v clang-18 || command -v clang-17 || command -v clang)
MARCH="rv64gcv_zfh"
FLAGS="-O2 -march=$MARCH -mabi=lp64d -ffp-contract=on --rtlib=compiler-rt"
# driver perf knobs: n=4096 (128 blocks/dot), 256 verify trials, 60000 timed iters best-of-7
DN=4096; DT=256; DI=60000
ROUNDS=12

echo "== ENV FINGERPRINT =="
echo "uname=$(uname -a)"
echo "clang=$C  $($C --version | head -1)"
echo "march=$MARCH  flags=$FLAGS"
echo "nproc=$(nproc)  pinned_core=$CORE"
echo "governor=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_governor 2>/dev/null)"
echo "cpu_max_khz=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/cpuinfo_max_freq 2>/dev/null)"
echo "libc=$(ls -l /lib/libc.so* 2>/dev/null | head -1; getconf GNU_LIBC_VERSION 2>/dev/null)"
echo "isa=$(grep -m1 -i isa /proc/cpuinfo)"

freq() { cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

# ---- compile: SAME clang/flags for our kernel AND factory ----
echo "== COMPILE =="
$C $FLAGS -x c++ "$OURK"           -c -o k_ours.o    2>cc_ours.err   && echo "k_ours.o OK"    || { echo "COMPILE_FAIL ours"; cat cc_ours.err; exit 20; }
$C $FLAGS -x c++ kernel_factory.c  -c -o k_factory.o 2>cc_factory.err&& echo "k_factory.o OK" || { echo "COMPILE_FAIL factory"; cat cc_factory.err; exit 21; }
$C $FLAGS -x c q8_0_verify_driver.c -c -o driver.o   2>cc_driver.err && echo "driver.o OK"    || { echo "COMPILE_FAIL driver"; cat cc_driver.err; exit 22; }
$C $FLAGS driver.o k_ours.o    -lm -o driver_ours    2>ld_ours.err   && echo "driver_ours OK"    || { echo "LINK_FAIL ours"; cat ld_ours.err; exit 23; }
$C $FLAGS driver.o k_factory.o -lm -o driver_factory 2>ld_factory.err&& echo "driver_factory OK" || { echo "LINK_FAIL factory"; cat ld_factory.err; exit 24; }
if [ "$DO_WINA" = "1" ]; then
  $C $FLAGS -x c++ kernel_m2.cpp -c -o k_m2.o 2>cc_m2.err && echo "k_m2.o OK" || { echo "COMPILE_FAIL m2"; cat cc_m2.err; }
  $C $FLAGS driver.o k_m2.o -lm -o driver_m2 2>ld_m2.err && echo "driver_m2 OK" || echo "LINK_FAIL m2"
fi

# ---- objdump seal (instruction封) ----
OD=$(command -v llvm-objdump-18 || command -v llvm-objdump-17 || command -v llvm-objdump || command -v objdump)
echo "== OBJDUMP SEAL ours ($LABEL) =="
$OD -d --no-show-raw-insn k_ours.o 2>/dev/null | sed -n '/tcrv_emitc/,/ret/p' | grep -E ':\s' | head -60
echo "== OBJDUMP SEAL factory =="
$OD -d --no-show-raw-insn k_factory.o 2>/dev/null | sed -n '/tcrv_emitc/,/ret/p' | grep -E ':\s' | head -60

# ---- correctness (bit-exact vs pinned oracle for ours; premul+fmaf for factory) ----
echo "== CORRECTNESS ours =="
taskset -c $CORE ./driver_ours $MARCH $DN $DT 2000 2>&1 | grep -E 'exact-match|VERDICT|property:|mutation caught|VLEN\(bits\)'
echo "== CORRECTNESS factory =="
taskset -c $CORE ./driver_factory $MARCH $DN $DT 2000 2>&1 | grep -E 'exact-match|VERDICT|property:|mutation caught'

# ---- paired A/B, 2 passes (pass2 => T-N between-run floor) ----
run_ns() { taskset -c $CORE "./$1" $MARCH $DN $DT $DI 2>/dev/null | awk '/^kernel/{print $3}'; }
# warmup (dropped)
taskset -c $CORE ./driver_ours $MARCH $DN $DT $DI >/dev/null 2>&1
taskset -c $CORE ./driver_factory $MARCH $DN $DT $DI >/dev/null 2>&1

for PASS in 1 2; do
  echo "== ABPASS $PASS =="
  for r in $(seq 1 $ROUNDS); do
    fo=$(freq); no=$(run_ns driver_ours)
    ff=$(freq); nf=$(run_ns driver_factory)
    echo "AB pass=$PASS round=$r side=ours    kernel_ns=$no freq_khz=$fo"
    echo "AB pass=$PASS round=$r side=factory kernel_ns=$nf freq_khz=$ff"
  done
done

# ---- winA (m1 vs m2, pure LMUL, both our pipeline) ----
if [ "$DO_WINA" = "1" ] && [ -x ./driver_m2 ]; then
  echo "== WINA (ours=$LABEL vs m2) =="
  taskset -c $CORE ./driver_m2 $MARCH $DN $DT $DI >/dev/null 2>&1
  for r in $(seq 1 $ROUNDS); do
    f1=$(freq); n1=$(run_ns driver_ours)
    f2=$(freq); n2=$(run_ns driver_m2)
    echo "WINA round=$r side=$LABEL kernel_ns=$n1 freq_khz=$f1"
    echo "WINA round=$r side=m2    kernel_ns=$n2 freq_khz=$f2"
  done
fi
echo "== DONE =="
